"""
ReWaterGAP Web API
Thin FastAPI wrapper around the existing run_watergap.py script.
"""

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # OnlineWaterGap/
JOBS_DIR = BASE_DIR / "jobs"
UPLOADS_DIR = BASE_DIR / "uploads"
PRESETS_DIR = Path(__file__).resolve().parent / "presets"
INPUT_DATA_DIR = BASE_DIR / "input_data"
JOBS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
PRESETS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="ReWaterGAP Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (single user, single process – good enough)
_current_job: Optional[dict] = None


# ---------------------------------------------------------------------------
# Pydantic models matching Config_ReWaterGAP.json structure
# ---------------------------------------------------------------------------
class FilePathInput(BaseModel):
    climate_forcing: str = "input_data/climate_forcing/"
    water_use_data: str = "input_data/water_use/"
    static_land_data: str = "input_data/static_input/"
    parameter_path: str = "model/WaterGAP_2.2e_global_parameters_gswp3_w5e5.nc"


class FilePathConfig(BaseModel):
    inputDir: FilePathInput = FilePathInput()
    outputDir: str = "output_data/"


class AntNatOpts(BaseModel):
    ant: bool = True
    subtract_use: bool = True
    res_opt: bool = True


class DemandSatisfactionOpts(BaseModel):
    delayed_use: bool = True
    neighbouring_cell: bool = True


class SimulationOption(BaseModel):
    AntNat_opts: AntNatOpts = AntNatOpts()
    Demand_satisfaction_opts: DemandSatisfactionOpts = DemandSatisfactionOpts()


class RestartOptions(BaseModel):
    restart: bool = False
    save_model_states_for_restart: bool = False
    save_and_read_states_dir: str = "./"


class SimulationPeriod(BaseModel):
    start: str = "1901-01-01"
    end: str = "1902-12-31"
    reservoir_start_year: int = 1901
    reservoir_end_year: int = 2019
    spinup_years: int = 5


class TimeStep(BaseModel):
    daily: bool = True


class SimulationExtent(BaseModel):
    run_basin: bool = False
    path_to_stations_file: str = "input_data/static_input/"


class CalibrateWaterGAP(BaseModel):
    run_calib: bool = False
    path_to_observed_discharge: str = "../test_wateruse/json_annual/"


class VerticalWaterBalanceFluxes(BaseModel):
    pot_evap: bool = False
    net_rad: bool = False
    leaf_area_index: bool = False
    canopy_evap: bool = False
    throughfall: bool = False
    snow_fall: bool = False
    snow_melt: bool = False
    snow_evap: bool = False
    groundwater_recharge_diffuse: bool = False
    surface_runoff: bool = False
    snowcover_frac: bool = False


class VerticalWaterBalanceStorages(BaseModel):
    canopy_storage: bool = False
    snow_water_equiv: bool = False
    soil_moisture: bool = False
    maximum_soil_moisture: bool = True


class LateralWaterBalanceFluxes(BaseModel):
    consistent_precipitation: bool = True
    groundwater_discharge: bool = False
    total_runoff: bool = False
    pot_cell_runoff: bool = False
    groundwater_recharge_swb: bool = False
    total_groundwater_recharge: bool = False
    local_lake_outflow: bool = False
    local_wetland_outflow: bool = False
    global_lake_outflow: bool = False
    global_wetland_outflow: bool = False
    streamflow: bool = True
    streamflow_from_upstream: bool = True
    actual_net_abstr_groundwater: bool = True
    actual_net_abstr_surfacewater: bool = True
    actual_water_consumption: bool = True
    cell_aet_consuse: bool = True
    unsat_potnetabs_sw_from_demandcell: bool = False
    returned_demand_from_supplycell: bool = False
    returned_demand_from_supplycell_nextday: bool = False
    demand_left_excl_returned_nextday: bool = False
    potnetabs_sw: bool = False
    get_neighbouring_cells_map: bool = False
    net_cell_runoff: bool = False
    river_velocity: bool = False
    land_area_fraction: bool = False


class LateralWaterBalanceStorages(BaseModel):
    groundwater_storage: bool = False
    local_lake_storage: bool = False
    local_wetland_storage: bool = False
    global_lake_storage: bool = False
    global_wetland_storage: bool = False
    river_storage: bool = False
    global_reservoir_storage: bool = False
    total_water_storage: bool = True


class SimulationConfig(BaseModel):
    """Full config matching Config_ReWaterGAP.json"""
    file_path_config: FilePathConfig = FilePathConfig()
    RuntimeOptions: Optional[list] = None
    OutputVariable: Optional[list] = None

    # Flat fields for the web form (converted to nested structure)
    simulation_option: Optional[SimulationOption] = None
    restart_options: Optional[RestartOptions] = None
    simulation_period: Optional[SimulationPeriod] = None
    time_step: Optional[TimeStep] = None
    simulation_extent: Optional[SimulationExtent] = None
    calibrate: Optional[CalibrateWaterGAP] = None
    vb_fluxes: Optional[VerticalWaterBalanceFluxes] = None
    vb_storages: Optional[VerticalWaterBalanceStorages] = None
    lb_fluxes: Optional[LateralWaterBalanceFluxes] = None
    lb_storages: Optional[LateralWaterBalanceStorages] = None

    def to_watergap_config(self) -> dict:
        """Convert to the nested JSON structure that ReWaterGAP expects."""
        # If RuntimeOptions/OutputVariable already provided (raw JSON mode), use as-is
        if self.RuntimeOptions is not None and self.OutputVariable is not None:
            return {
                "FilePath": self.file_path_config.model_dump(),
                "RuntimeOptions": self.RuntimeOptions,
                "OutputVariable": self.OutputVariable,
            }

        # Build from flat fields (web form mode)
        sim_opt = self.simulation_option or SimulationOption()
        restart = self.restart_options or RestartOptions()
        period = self.simulation_period or SimulationPeriod()
        ts = self.time_step or TimeStep()
        extent = self.simulation_extent or SimulationExtent()
        calib = self.calibrate or CalibrateWaterGAP()

        vb_f = self.vb_fluxes or VerticalWaterBalanceFluxes()
        vb_s = self.vb_storages or VerticalWaterBalanceStorages()
        lb_f = self.lb_fluxes or LateralWaterBalanceFluxes()
        lb_s = self.lb_storages or LateralWaterBalanceStorages()

        return {
            "FilePath": self.file_path_config.model_dump(),
            "RuntimeOptions": [
                {"SimulationOption": sim_opt.model_dump()},
                {"RestartOptions": restart.model_dump()},
                {"SimulationPeriod": period.model_dump()},
                {"TimeStep": ts.model_dump()},
                {"SimulationExtent": extent.model_dump()},
                {"Calibrate WaterGAP": calib.model_dump()},
            ],
            "OutputVariable": [
                {"VerticalWaterBalanceFluxes": vb_f.model_dump()},
                {"VerticalWaterBalanceStorages": vb_s.model_dump()},
                {"LateralWaterBalanceFluxes": lb_f.model_dump()},
                {"LateralWaterBalanceStorages": lb_s.model_dump()},
            ],
        }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _get_job_status(job: dict) -> dict:
    """Check process status and read log tail."""
    proc: subprocess.Popen = job.get("process")
    log_file = Path(job["log_path"])

    # Check if process is still running
    if proc and proc.poll() is None:
        status = "running"
    elif proc and proc.returncode == 0:
        status = "completed"
        # Close log file handle when process finishes
        log_fh = job.get("log_fh")
        if log_fh and not log_fh.closed:
            log_fh.close()
    elif proc:
        status = "failed"
        log_fh = job.get("log_fh")
        if log_fh and not log_fh.closed:
            log_fh.close()
    else:
        status = "unknown"

    # Read last 50 lines of log
    log_tail = ""
    if log_file.exists():
        lines = log_file.read_text(errors="replace").splitlines()
        log_tail = "\n".join(lines[-50:])

    return {
        "job_id": job["job_id"],
        "status": status,
        "started_at": job["started_at"],
        "config_summary": {
            "start": job.get("config_summary", {}).get("start", ""),
            "end": job.get("config_summary", {}).get("end", ""),
        },
        "log_tail": log_tail,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/schema")
def get_schema():
    """Return JSON schema for the simulation config (for frontend validation)."""
    return SimulationConfig.model_json_schema()


@app.get("/api/presets")
def get_presets():
    """Return list of preset configurations."""
    presets = []

    # Always include the default config
    default_config_path = BASE_DIR / "Config_ReWaterGAP.json"
    if default_config_path.exists():
        with open(default_config_path) as f:
            presets.append({
                "id": "default",
                "name": "Standard (from repository)",
                "description": "The original configuration from Config_ReWaterGAP.json",
                "config": json.load(f),
            })

    # Load any custom presets
    for preset_file in sorted(PRESETS_DIR.glob("*.json")):
        with open(preset_file) as f:
            data = json.load(f)
            presets.append({
                "id": preset_file.stem,
                "name": data.get("_name", preset_file.stem),
                "description": data.get("_description", ""),
                "config": {k: v for k, v in data.items()
                           if not k.startswith("_")},
            })

    return presets


@app.get("/api/input-datasets")
def list_input_datasets():
    """List available input datasets grouped by category.

    Scans input_data/ for climate_forcing, water_use, and static_input
    subdirectories and returns the files found in each.
    """
    categories = {
        "climate_forcing": {
            "label": "Climate Forcing",
            "description": "Meteorological driving data (precipitation, temperature, radiation, etc.)",
            "path": "input_data/climate_forcing/",
        },
        "water_use": {
            "label": "Water Use",
            "description": "Sectoral water withdrawal and consumption data",
            "path": "input_data/water_use/",
        },
        "static_input": {
            "label": "Static Land Data",
            "description": "Land cover, soil, routing, and other static parameters",
            "path": "input_data/static_input/",
        },
    }

    datasets = []

    # Check if input_data directory exists at all
    if not INPUT_DATA_DIR.exists():
        return {"datasets": [], "custom_upload_enabled": True}

    # Build the default dataset bundle from what's on disk
    bundle_files = {}
    total_size_mb = 0
    all_present = True

    for cat_key, cat_info in categories.items():
        cat_path = BASE_DIR / cat_info["path"]
        files = []
        if cat_path.exists():
            for f in sorted(cat_path.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                    total_size_mb += size_mb
                    files.append({
                        "name": f.name,
                        "size_mb": size_mb,
                    })
        else:
            all_present = False

        bundle_files[cat_key] = {
            "label": cat_info["label"],
            "description": cat_info["description"],
            "files": files,
            "available": len(files) > 0,
        }

    if any(cat["available"] for cat in bundle_files.values()):
        datasets.append({
            "id": "gswp3-w5e5",
            "name": "GSWP3-W5E5 (Default)",
            "description": "Standard dataset: GSWP3-W5E5 climate forcing with matching water use data",
            "total_size_mb": round(total_size_mb, 1),
            "complete": all_present,
            "categories": bundle_files,
            "paths": {
                "climate_forcing": "input_data/climate_forcing/",
                "water_use": "input_data/water_use/",
                "static_input": "input_data/static_input/",
            },
        })

    return {"datasets": datasets, "custom_upload_enabled": True}


@app.post("/api/simulate")
def start_simulation(config: SimulationConfig):
    """Start a new simulation. Only one at a time."""
    global _current_job

    # Check if a simulation is already running
    if _current_job:
        proc = _current_job.get("process")
        if proc and proc.poll() is None:
            raise HTTPException(
                status_code=409,
                detail="A simulation is already running. Please wait or cancel it."
            )

    # Create job directory
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # Write config file
    watergap_config = config.to_watergap_config()
    config_path = job_dir / "Config_ReWaterGAP.json"
    with open(config_path, "w") as f:
        json.dump(watergap_config, f, indent=2)

    # Set output dir to job-specific folder
    output_dir = job_dir / "output"
    output_dir.mkdir()
    watergap_config["FilePath"]["outputDir"] = str(output_dir) + "/"
    with open(config_path, "w") as f:
        json.dump(watergap_config, f, indent=2)

    # Start subprocess
    log_path = job_dir / "simulation.log"
    env = os.environ.copy()
    env["WATERGAP_CONFIG"] = str(config_path)
    env["PYTHONUNBUFFERED"] = "1"

    log_fh = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "run_watergap.py"],
            cwd=str(BASE_DIR),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        log_fh.close()
        shutil.rmtree(str(job_dir), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error starting simulation: {e}")

    # Extract summary from config for display
    period = {}
    try:
        period = watergap_config["RuntimeOptions"][2]["SimulationPeriod"]
    except (IndexError, KeyError):
        pass

    _current_job = {
        "job_id": job_id,
        "process": proc,
        "log_path": str(log_path),
        "log_fh": log_fh,
        "job_dir": str(job_dir),
        "started_at": datetime.now().isoformat(),
        "config_summary": {
            "start": period.get("start", ""),
            "end": period.get("end", ""),
        },
    }

    return {"job_id": job_id, "status": "started"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    """Get status of a simulation job."""
    if not _current_job or _current_job["job_id"] != job_id:
        # Check if job dir exists (completed earlier)
        job_dir = JOBS_DIR / job_id
        if job_dir.exists():
            log_path = job_dir / "simulation.log"
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-50:])
            output_dir = job_dir / "output"
            has_results = output_dir.exists() and any(output_dir.iterdir())
            return {
                "job_id": job_id,
                "status": "completed" if has_results else "failed",
                "log_tail": log_tail,
            }
        raise HTTPException(status_code=404, detail="Job not found")

    return _get_job_status(_current_job)


@app.post("/api/cancel/{job_id}")
def cancel_simulation(job_id: str):
    """Cancel a running simulation."""
    global _current_job
    if not _current_job or _current_job["job_id"] != job_id:
        raise HTTPException(status_code=404, detail="Job not found")

    proc = _current_job.get("process")
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return {"status": "cancelled"}
    return {"status": "not_running"}


@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    """Download simulation results as ZIP."""
    job_dir = JOBS_DIR / job_id
    output_dir = job_dir / "output"

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="No results found")

    # Check if there are actual output files
    output_files = list(output_dir.glob("*"))
    if not output_files:
        raise HTTPException(status_code=404, detail="No result files available")

    # Create ZIP
    zip_path = job_dir / "results"
    shutil.make_archive(str(zip_path), "zip", str(output_dir))

    return FileResponse(
        path=str(zip_path) + ".zip",
        filename=f"rewatergap_results_{job_id}.zip",
        media_type="application/zip",
    )


@app.get("/api/log/{job_id}")
def stream_log(job_id: str):
    """Stream the full log file."""
    job_dir = JOBS_DIR / job_id
    log_path = job_dir / "simulation.log"

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")

    return FileResponse(path=str(log_path), media_type="text/plain")


# ---------------------------------------------------------------------------
# Upload endpoints (for custom NetCDF viewing)
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs():
    """List all completed simulation jobs."""
    jobs = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            output_dir = d / "output"
            if output_dir.exists() and any(output_dir.glob("*.nc")):
                # Read config summary if available
                config_path = d / "Config_ReWaterGAP.json"
                period_str = ""
                if config_path.exists():
                    try:
                        cfg = json.loads(config_path.read_text())
                        p = cfg.get("RuntimeOptions", [{}])[2].get("SimulationPeriod", {})
                        period_str = f"{p.get('start', '')} to {p.get('end', '')}"
                    except Exception:
                        pass
                jobs.append({
                    "job_id": d.name,
                    "period": period_str,
                })
    return jobs


@app.post("/api/upload")
async def upload_netcdf(file: UploadFile = File(...)):
    """Upload a NetCDF file for visualization."""
    if not file.filename.endswith(".nc"):
        raise HTTPException(status_code=400, detail="Only .nc (NetCDF) files are accepted")

    upload_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    upload_dir = UPLOADS_DIR / upload_id
    upload_dir.mkdir(parents=True)

    dest = upload_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)

    # Validate it's a real NetCDF file
    try:
        with xr.open_dataset(dest) as ds:
            var_names = list(ds.data_vars)
            if not var_names:
                shutil.rmtree(str(upload_dir), ignore_errors=True)
                raise HTTPException(status_code=400, detail="NetCDF file contains no data variables")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        shutil.rmtree(str(upload_dir), ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Invalid NetCDF file: {e}")

    return {"upload_id": upload_id, "filename": file.filename, "variables": var_names}


@app.get("/api/uploads")
def list_uploads():
    """List all uploaded NetCDF datasets."""
    uploads = []
    for d in sorted(UPLOADS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            nc_files = list(d.glob("*.nc"))
            if nc_files:
                uploads.append({
                    "upload_id": d.name,
                    "filename": nc_files[0].name,
                    "uploaded_at": d.name[:15].replace("_", " ", 1),
                })
    return uploads


@app.delete("/api/upload/{upload_id}")
def delete_upload(upload_id: str):
    """Delete an uploaded dataset."""
    upload_dir = UPLOADS_DIR / upload_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    shutil.rmtree(str(upload_dir), ignore_errors=True)
    return {"status": "deleted"}


def _resolve_output_dir(source_id: str) -> Path:
    """Resolve an output directory from a job ID or upload ID."""
    # Check jobs first
    job_dir = JOBS_DIR / source_id / "output"
    if job_dir.exists():
        return job_dir
    # Check uploads
    upload_dir = UPLOADS_DIR / source_id
    if upload_dir.exists():
        return upload_dir
    raise HTTPException(status_code=404, detail="Source not found")


# ---------------------------------------------------------------------------
# Visualization endpoints
# ---------------------------------------------------------------------------

# Human-readable labels + units for known variables
_VAR_META = {
    "dis": {"label": "Streamflow (Discharge)", "unit": "m³/s", "cmap": "YlGnBu"},
    "dis-from-upstream": {"label": "Streamflow from Upstream", "unit": "m³/s", "cmap": "YlGnBu"},
    "tws": {"label": "Total Water Storage (TWS)", "unit": "mm", "cmap": "BrBG"},
    "consistent-precipitation": {"label": "Precipitation", "unit": "mm/d", "cmap": "YlGnBu"},
    "evap-total": {"label": "Evapotranspiration", "unit": "mm/d", "cmap": "YlOrRd"},
    "atotuse": {"label": "Total Water Use", "unit": "m³/s", "cmap": "Reds"},
    "atotusegw": {"label": "Groundwater Use", "unit": "m³/s", "cmap": "Oranges"},
    "atotusesw": {"label": "Surface Water Use", "unit": "m³/s", "cmap": "Blues"},
    "smax": {"label": "Max. Soil Moisture Storage", "unit": "mm", "cmap": "YlGn"},
}


def _collect_output_files(output_dir: Path) -> dict:
    """Group NetCDF files by variable name, return metadata."""
    variables = {}
    for nc in sorted(output_dir.glob("*.nc")):
        # Pattern: varname_YYYY-MM-DD.nc or varname.nc (static)
        stem = nc.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and len(parts[1]) == 10:  # e.g. dis_1901-12-31
            var_name = parts[0]
        else:
            var_name = stem  # static file like smax.nc

        if var_name not in variables:
            variables[var_name] = {"files": [], "static": False}
        variables[var_name]["files"].append(nc.name)

    # Detect static vs temporal and read time steps
    for var_name, info in variables.items():
        if len(info["files"]) == 1 and "_" not in Path(info["files"][0]).stem:
            info["static"] = True
            info["time_steps"] = []
        else:
            # Read actual time coords from first file
            first_file = output_dir / info["files"][0]
            try:
                with xr.open_dataset(first_file) as ds:
                    if "time" in ds.coords:
                        times = [str(t)[:10] for t in ds.time.values]
                        info["time_steps"] = times
                    else:
                        info["time_steps"] = []
                        info["static"] = True
            except Exception:
                info["time_steps"] = []

        # Add human-readable metadata
        meta = _VAR_META.get(var_name, {})
        info["label"] = meta.get("label", var_name)
        info["unit"] = meta.get("unit", "")
        info["cmap"] = meta.get("cmap", "viridis")

    return variables


@app.get("/api/variables/{source_id}")
def get_variables(source_id: str):
    """List all output variables with time steps for a job or upload."""
    output_dir = _resolve_output_dir(source_id)

    variables = _collect_output_files(output_dir)

    # Build response: for each variable, list all time steps across files
    result = {}
    for var_name, info in variables.items():
        if info["static"]:
            all_times = []
        else:
            # Collect time steps from all files for this variable
            all_times = []
            for fname in info["files"]:
                try:
                    with xr.open_dataset(output_dir / fname) as ds:
                        if "time" in ds.coords:
                            all_times.extend([str(t)[:10] for t in ds.time.values])
                except Exception:
                    pass

        result[var_name] = {
            "label": info["label"],
            "unit": info["unit"],
            "static": info["static"],
            "time_steps": all_times,
            "files": info["files"],
        }

    return result


@app.get("/api/map/{source_id}")
def get_map_image(
    source_id: str,
    var: str = Query(..., description="Variable name"),
    time: Optional[str] = Query(None, description="Date YYYY-MM-DD"),
    cmap: Optional[str] = Query(None, description="Colormap override"),
    width: int = Query(12, ge=4, le=20),
    height: int = Query(6, ge=2, le=12),
    vmin: Optional[float] = Query(None),
    vmax: Optional[float] = Query(None),
    log_scale: bool = Query(False),
):
    """Render a map image (PNG) for a given variable and time step."""
    output_dir = _resolve_output_dir(source_id)

    # Find the right NetCDF file
    matching_files = sorted(output_dir.glob(f"{var}*.nc"))
    if not matching_files:
        raise HTTPException(status_code=404, detail=f"Variable '{var}' not found")

    # For static variables, just open the single file
    if len(matching_files) == 1 and "_" not in matching_files[0].stem:
        ds = xr.open_dataset(matching_files[0])
        data_var = list(ds.data_vars)[0]
        data = ds[data_var]
        title_time = ""
    else:
        # Find the file containing the requested date
        target_ds = None
        if time is None:
            # Default: first time step of first file
            ds = xr.open_dataset(matching_files[0])
            data_var = list(ds.data_vars)[0]
            data = ds[data_var].isel(time=0)
            title_time = str(ds.time.values[0])[:10]
        else:
            # Search across files for the requested date
            found = False
            for nc_file in matching_files:
                ds = xr.open_dataset(nc_file)
                data_var = list(ds.data_vars)[0]
                if "time" in ds.coords:
                    time_strs = [str(t)[:10] for t in ds.time.values]
                    if time in time_strs:
                        idx = time_strs.index(time)
                        data = ds[data_var].isel(time=idx)
                        title_time = time
                        found = True
                        break
                ds.close()
            if not found:
                raise HTTPException(
                    status_code=404,
                    detail=f"Time step '{time}' not found for variable '{var}'"
                )

    # Get metadata
    meta = _VAR_META.get(var, {})
    colormap = cmap or meta.get("cmap", "viridis")
    unit = meta.get("unit", "")
    label = meta.get("label", var)

    # Convert to numpy, replace 0 with NaN for better visualization (water vars)
    values = data.values.copy()
    if var in ("dis", "dis-from-upstream", "atotuse", "atotusegw", "atotusesw"):
        values[values == 0] = np.nan

    # Auto-scale: use 2nd–98th percentile to avoid outliers
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        raise HTTPException(status_code=404, detail="No valid data")

    auto_vmin = float(np.nanpercentile(valid, 2))
    auto_vmax = float(np.nanpercentile(valid, 98))
    plot_vmin = vmin if vmin is not None else auto_vmin
    plot_vmax = vmax if vmax is not None else auto_vmax

    # Build the figure
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    if log_scale and plot_vmin > 0:
        norm = mcolors.LogNorm(vmin=max(plot_vmin, 1e-6), vmax=plot_vmax)
    else:
        norm = mcolors.Normalize(vmin=plot_vmin, vmax=plot_vmax)

    lat = data.lat.values if hasattr(data, "lat") else np.arange(values.shape[0])
    lon = data.lon.values if hasattr(data, "lon") else np.arange(values.shape[1])

    im = ax.pcolormesh(
        lon, lat, values,
        cmap=colormap, norm=norm, shading="auto", rasterized=True,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add coastlines via simple rectangle hint (lightweight, no cartopy needed)
    ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.3, alpha=0.5)

    # Colorbar
    cbar_label = f"{label} [{unit}]" if unit else label
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.8, pad=0.02)

    # Title
    title = label
    if title_time:
        title += f"  —  {title_time}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    ds.close()
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Serve frontend (must be LAST – catches all remaining routes)
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

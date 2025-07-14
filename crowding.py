import os, json, pathlib, requests
from rapidfuzz import process, fuzz           # pip install rapidfuzz≠1

APP_KEY = os.getenv("TFL_APP_KEY")

_SESSION = requests.Session()

CACHE   = pathlib.Path.home() / ".cache/tfl_tube_stations.json"

def _refresh_cache() -> list[dict]:
    params = {
        "modes": "tube",
        "stopType": "NaptanMetroStation",
        "useStopPointHierarchy": "false",
        "app_key": APP_KEY,
    }

    url     = "https://api.tfl.gov.uk/StopPoint/Mode/tube/"

    data = _SESSION.get(url, params=params, timeout=60).json()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2))
    return data

def _load_stations(force_update: bool = False) -> list[dict]:
    if force_update or not CACHE.exists():
        return _refresh_cache()
    return json.loads(CACHE.read_text())['stopPoints']

_STATIONS = _load_stations()

def list_station_names() -> list[str]:
    """Alphabetical ‘Oxford Circus’, ‘Ealing Broadway’, …"""
    return sorted(sp["commonName"] for sp in _STATIONS)

def best_station_match(user_text: str, *, min_score: int = 80) -> tuple[str, str] | None:
    """
    Return (name, naptanId) for the closest match, or None if below threshold.
    tweak `processor` / `scorer` in `process.extractOne` for different heuristics.
    """
    names = {sp["commonName"]: sp["naptanId"] for sp in _STATIONS}
    matches = process.extract(user_text, names.keys(), scorer=fuzz.WRatio, limit = 10)
    for match in matches:
        if match and (match[1] >= min_score) and names[match[0]].startswith("940G"):
            return match[0], names[match[0]]
    return None

def _naptan_for_station(name: str) -> str:
    """
    Resolve a human-readable Underground station name to its NaPTAN code
    using /StopPoint/Search.  Raises ValueError if no Tube station is found.
    """
    url = f"https://api.tfl.gov.uk/StopPoint/Search/{name}"
    params = {"modes": "tube", "app_key": APP_KEY}

    r = _SESSION.get(url, params=params, timeout=5)
    r.raise_for_status()
    for match in r.json().get("matches", []):
        nid = match["id"]
        # Tube stations always start 940G…  (Bakerloo = 940GZZLU… etc.)
        if nid.startswith("940G"):
            return nid

    raise ValueError(f"No Tube station called “{name}” was found")

def live_crowding(station_name: str) -> dict:
    """
    Return the latest crowd-level payload for a London Underground station.
    Example field:  {'percentageOfBaseline': 0.31, 'timeUtc': '2025-07-14T09:30:00Z', …}
    """
    naptan = _naptan_for_station(station_name)
    url    = f"https://api.tfl.gov.uk/crowding/{naptan}/Live"
    params = {"app_key": APP_KEY}

    r = _SESSION.get(url, params=params, timeout=5)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    print(live_crowding(best_station_match("Holborn")[0]))

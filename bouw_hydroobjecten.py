# %%
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import force_2d
from shapely.geometry import LineString, Point
from shapely.ops import substring
from geopandas import GeoDataFrame
from pathlib import Path

DATA_DIR = Path(__file__).parent.joinpath("data")
GEOMETRIES_ALLOWED = ["LineString"]

# input on cloud
fairway_osm_path = DATA_DIR.joinpath("waterway_fairway.gpkg")
river_osm_path = DATA_DIR.joinpath("waterway_river.gpkg")
canal_osm_path = DATA_DIR.joinpath("waterway_canal.gpkg")
extra_lines_path = DATA_DIR.joinpath("extra_lijnen.gpkg")

# input from previous step
basins_path = DATA_DIR.joinpath("basins.gpkg")

# output
hydamo_path = DATA_DIR.joinpath("hydamo.gpkg")


def snap_boundaries_to_other_line(
    line: LineString,
    start_line: LineString | None,
    end_line: LineString | None,
    tolerance: float = 0.1,
) -> LineString | None:
    """Snap line boundaries to nearby reference lines when available."""
    if line.is_empty:
        return None

    line_boundary = tuple(line.boundary.geoms)
    if len(line_boundary) != 2:
        return None

    coords = list(line.coords)
    start_pt, end_pt = line_boundary

    if start_line is not None:
        start_boundary = tuple(start_line.boundary.geoms)
        if len(start_boundary) != 2 or start_pt.distance(start_line) >= tolerance:
            return None
        if start_pt.distance(start_line.boundary) < tolerance:
            pt = min(start_boundary, key=start_pt.distance)
        else:
            pt = start_line.interpolate(start_line.project(start_pt))
        coords[0] = (pt.x, pt.y)

    if end_line is not None:
        end_boundary = tuple(end_line.boundary.geoms)
        if len(end_boundary) != 2 or end_pt.distance(end_line) >= tolerance:
            return None
        if end_pt.distance(end_line.boundary) < tolerance:
            pt = min(end_boundary, key=end_pt.distance)
        else:
            pt = end_line.interpolate(end_line.project(end_pt))
        coords[-1] = (pt.x, pt.y)

    snapped_line = LineString(coords)
    if snapped_line.length == 0 or len(tuple(snapped_line.boundary.geoms)) != 2:
        return None

    return snapped_line


def split_line_at_distances(
    line: LineString, distances: list[float]
) -> list[LineString]:
    """Split a line at projected distances along the line."""
    if not distances:
        return [line]

    split_distances = []
    line_length = line.length
    for distance in sorted(distances):
        if distance <= 0 or distance >= line_length:
            continue
        if split_distances and np.isclose(distance, split_distances[-1]):
            continue
        split_distances.append(distance)

    if not split_distances:
        return [line]

    segments = []
    start_distance = 0.0
    for end_distance in [*split_distances, line_length]:
        segment = substring(line, start_distance, end_distance)
        if isinstance(segment, LineString) and segment.length > 0:
            segments.append(segment)
        start_distance = end_distance

    return segments or [line]


def split_lines_at_nearby_boundaries(
    gdf: GeoDataFrame, tolerance: float
) -> GeoDataFrame:
    """Split lines where another line boundary lies on the line within tolerance."""
    source_geometries = gdf.geometry.to_list()
    spatial_index = gdf.sindex
    records = []

    for row in gdf.itertuples(index=False):
        line = row.geometry
        if line.is_empty or len(tuple(line.boundary.geoms)) != 2:
            continue

        minx, miny, maxx, maxy = line.bounds
        candidate_indices = spatial_index.intersection(
            (minx - tolerance, miny - tolerance, maxx + tolerance, maxy + tolerance)
        )
        split_distances = []

        for other_idx in candidate_indices:
            other_line = source_geometries[other_idx]
            if (
                other_line is line
                or other_line.is_empty
                or len(tuple(other_line.boundary.geoms)) != 2
            ):
                continue

            for boundary_pt in other_line.boundary.geoms:
                projected_distance = line.project(boundary_pt)
                if projected_distance <= tolerance or projected_distance >= (
                    line.length - tolerance
                ):
                    continue

                projected_point = line.interpolate(projected_distance)
                if boundary_pt.distance(projected_point) < tolerance:
                    split_distances.append(projected_distance)

        row_data = row._asdict()
        for segment in split_line_at_distances(line, split_distances):
            row_data["geometry"] = segment
            records.append(row_data.copy())

    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)


def snap_line_boundaries(
    gdf: GeoDataFrame, tolerance: float, basin_union
) -> GeoDataFrame:
    """Snap the boundaries of a linestring geodataframe to the other boundaries, or lines within the set that are within tolerance"""
    snapped_gdf = gdf.copy()
    source_geometries = snapped_gdf.geometry.to_list()
    geometries = source_geometries.copy()
    spatial_index = gdf.sindex

    for line_idx, line in enumerate(source_geometries):
        if line is None or line.is_empty or len(tuple(line.boundary.geoms)) != 2:
            geometries[line_idx] = None
            continue

        minx, miny, maxx, maxy = line.bounds
        candidate_indices = spatial_index.intersection(
            (minx - tolerance, miny - tolerance, maxx + tolerance, maxy + tolerance)
        )
        start_pt, end_pt = tuple(line.boundary.geoms)
        start_match = None
        end_match = None
        start_distance = tolerance
        end_distance = tolerance

        for other_idx in candidate_indices:
            if other_idx == line_idx:
                continue

            other_line = source_geometries[other_idx]
            if (
                other_line is None
                or other_line.is_empty
                or len(tuple(other_line.boundary.geoms)) != 2
                or not other_line.within(basin_union)
            ):
                continue

            start_other_distance = start_pt.distance(other_line)
            if start_other_distance < start_distance:
                start_match = other_line
                start_distance = start_other_distance

            end_other_distance = end_pt.distance(other_line)
            if end_other_distance < end_distance:
                end_match = other_line
                end_distance = end_other_distance

        match_count = int(start_match is not None) + int(end_match is not None)
        if match_count == 0:
            geometries[line_idx] = None
            continue
        if match_count == 1 and not line.within(basin_union):
            geometries[line_idx] = None
            continue

        geometries[line_idx] = snap_boundaries_to_other_line(
            line=line,
            start_line=start_match,
            end_line=end_match,
            tolerance=tolerance,
        )

    snapped_gdf.loc[:, "geometry"] = geometries
    snapped_gdf = snapped_gdf[snapped_gdf.geometry.notna()].copy()
    return snapped_gdf


# %% read files

print("read basins")
basins_gdf = gpd.read_file(basins_path, layer="ribasim_basins")

print("read osm fairway")
fairway_osm_gdf = gpd.read_file(fairway_osm_path, fid_as_index=True)

print("read osm river")
river_osm_gdf = gpd.read_file(river_osm_path, fid_as_index=True)

print("read osm canals")
canal_osm_gdf = gpd.read_file(canal_osm_path, fid_as_index=True)

print("read extra lijnen")
extra_lines_gdf = gpd.read_file(extra_lines_path, fid_as_index=True)

# %% aanmaken masks

# osm_basins voor het filteren van osm_lijnen
print("samenstellen clip-polygonen")
ijsselmeer_basins = [
    "Markermeer",
    "Gouwzee",
    "IJmeer",
    "IJsselmeer",
]  # ijsselmeer komt uit extra lijnen
osm_basins_gdf = basins_gdf[~basins_gdf["naam"].isin(ijsselmeer_basins)]
ijsselmeer_poly = basins_gdf[basins_gdf["naam"].isin(ijsselmeer_basins)].union_all()

# samenvoegen van alle OSM lijnen
network_lines_gdf = pd.concat(
    [
        river_osm_gdf,
        canal_osm_gdf,
        fairway_osm_gdf,
    ],
    ignore_index=True,
)
network_lines_gdf.loc[:, ["original_index"]] = network_lines_gdf.index + 1

print("osm clippen op polygonen")
osm_basins_with_order_gdf = osm_basins_gdf[["geometry"]].copy()
osm_basins_with_order_gdf["basin_order"] = np.arange(len(osm_basins_with_order_gdf))
matched_lines_gdf = gpd.sjoin(
    network_lines_gdf[["original_index", "geometry"]],
    osm_basins_with_order_gdf,
    how="inner",
    predicate="intersects",
)
matched_original_indices = (
    matched_lines_gdf.reset_index(names="line_order")
    .sort_values(["basin_order", "line_order"])
    .drop_duplicates("original_index")["original_index"]
)
lines_gdf = (
    network_lines_gdf.set_index("original_index")
    .loc[matched_original_indices]
    .reset_index()
)


# %%
print("osm lijnen samenvoegen met extra lijnen")
extra_lines_gdf.rename(columns={"naam": "name"}, inplace=True)
lines_gdf.rename(columns={"osm_id": "id"}, inplace=True)
lines_gdf = pd.concat(
    [
        lines_gdf,
        extra_lines_gdf,
    ],
    ignore_index=True,
)


# Assuming network_lines_gdf is defined somewhere before this point
lines_gdf = lines_gdf[
    ~lines_gdf["name"].isin(["Geul", "Derde Diem"])
]  # brute verwijdering wegens sifon onder Julianakanaal


lines_gdf = lines_gdf.explode(index_parts=False).copy()

geom_types = lines_gdf.geom_type.unique()
if not all(geom_type in GEOMETRIES_ALLOWED for geom_type in geom_types):
    raise ValueError(
        f"Only geom_types {GEOMETRIES_ALLOWED} are allowed. Got {geom_types}"
    )

lines_gdf.loc[:, "geometry"] = lines_gdf.geometry.apply(force_2d)
lines_gdf = lines_gdf[lines_gdf.boundary.count_geometries() == 2].copy()

lines_gdf = pd.concat(
    [
        lines_gdf[lines_gdf["full_id"].isna()],
        lines_gdf[lines_gdf["full_id"].notna()].drop_duplicates("full_id"),
    ],
    ignore_index=True,
)

lines_gdf = split_lines_at_nearby_boundaries(lines_gdf, tolerance=0.25)
basin_union = basins_gdf.union_all()
lines_gdf = snap_line_boundaries(lines_gdf, tolerance=0.25, basin_union=basin_union)

print("write to hydamo")
lines_gdf.rename(columns={"name": "naam"}, inplace=True)
lines_gdf.to_file(hydamo_path, layer="hydroobject")

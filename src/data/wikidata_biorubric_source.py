from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.constants import BIORUBRIC_ALLOWED_SLOTS, BIORUBRIC_SLOT_PROPERTY_IDS

WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_USER_AGENT = "criteria-over-beliefs-build/0.1 (research automation)"
WIKIDATA_MAX_SEARCH_RESULTS = 50
WIKIDATA_MAX_ENTITY_IDS_PER_REQUEST = 50


class WikidataBioRubricSourceError(ValueError):
    """Raised when BioRubric truthy-source extraction fails."""


def _extract_entity_id(entity_url: str) -> str:
    prefix = "http://www.wikidata.org/entity/"
    secure_prefix = "https://www.wikidata.org/entity/"
    if entity_url.startswith(prefix):
        return entity_url[len(prefix) :]
    if entity_url.startswith(secure_prefix):
        return entity_url[len(secure_prefix) :]
    raise WikidataBioRubricSourceError(f"Expected a Wikidata entity URL, got {entity_url!r}.")


def _search_query_for_slot(property_id: str) -> str:
    return f"haswbstatement:P31=Q5 haswbstatement:P106 haswbstatement:{property_id}"


def _request_json(url: str, *, timeout_seconds: int, max_retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        request = urllib.request.Request(url, headers={"User-Agent": WIKIDATA_USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:  # pragma: no cover - network failures vary by env
            last_error = error
            if error.code == 429:
                retry_after = error.headers.get("Retry-After")
                try:
                    sleep_seconds = float(retry_after) if retry_after is not None else 15.0
                except ValueError:
                    sleep_seconds = 15.0
                time.sleep(max(sleep_seconds, 15.0))
                continue
            if error.code >= 500 and attempt < max_retries - 1:
                time.sleep(min(2**attempt, 8))
                continue
            break
        except Exception as error:  # pragma: no cover - network failures vary by env
            last_error = error
            if attempt == max_retries - 1:
                break
            time.sleep(min(2**attempt, 8))

    raise WikidataBioRubricSourceError(f"Wikidata request failed after retries: {last_error}")


def _request_api_json(
    params: Mapping[str, Any],
    *,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    encoded = urllib.parse.urlencode({**params, "format": "json"})
    return _request_json(
        f"{WIKIDATA_API_ENDPOINT}?{encoded}",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _request_sparql_json(
    query: str,
    *,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    encoded_query = urllib.parse.quote(query)
    return _request_json(
        f"{WIKIDATA_SPARQL_ENDPOINT}?format=json&query={encoded_query}",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _label_from_entity_payload(entity_payload: Mapping[str, Any]) -> str | None:
    labels = entity_payload.get("labels")
    if not isinstance(labels, Mapping):
        return None

    preferred = labels.get("en")
    if isinstance(preferred, Mapping):
        value = preferred.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()

    for label_payload in labels.values():
        if not isinstance(label_payload, Mapping):
            continue
        value = label_payload.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _claim_target_entity_id(claim: Mapping[str, Any]) -> str | None:
    mainsnak = claim.get("mainsnak")
    if not isinstance(mainsnak, Mapping):
        return None
    if mainsnak.get("snaktype") != "value":
        return None

    datavalue = mainsnak.get("datavalue")
    if not isinstance(datavalue, Mapping):
        return None
    value = datavalue.get("value")
    if not isinstance(value, Mapping):
        return None

    entity_id = value.get("id")
    if isinstance(entity_id, str) and entity_id:
        return entity_id

    numeric_id = value.get("numeric-id")
    if isinstance(numeric_id, int):
        return f"Q{numeric_id}"
    return None


def _extract_truthy_entity_ids_from_claims(
    claims: Mapping[str, Any],
    property_id: str,
) -> list[str]:
    claim_list = claims.get(property_id, [])
    if not isinstance(claim_list, list):
        return []

    preferred_ids: list[str] = []
    normal_ids: list[str] = []
    for claim in claim_list:
        if not isinstance(claim, Mapping):
            continue
        rank = claim.get("rank")
        if rank == "deprecated":
            continue
        target_id = _claim_target_entity_id(claim)
        if target_id is None:
            continue
        if rank == "preferred":
            preferred_ids.append(target_id)
        elif rank == "normal":
            normal_ids.append(target_id)

    truthy_ids = preferred_ids if preferred_ids else normal_ids
    return sorted(set(truthy_ids))


def _fetch_search_entity_ids(
    slot: str,
    *,
    page_size: int,
    max_pages: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[str]:
    if slot not in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}:
        raise WikidataBioRubricSourceError("Search fetches must target one distinguishing slot.")
    if page_size <= 0 or max_pages <= 0:
        raise WikidataBioRubricSourceError("page_size and max_pages must be positive integers.")

    property_id = BIORUBRIC_SLOT_PROPERTY_IDS[slot]
    search_limit = min(page_size, WIKIDATA_MAX_SEARCH_RESULTS)
    offset = 0
    entity_ids: list[str] = []

    for page_index in range(max_pages):
        payload = _request_api_json(
            {
                "action": "query",
                "list": "search",
                "srsearch": _search_query_for_slot(property_id),
                "srnamespace": "0",
                "srlimit": str(search_limit),
                "sroffset": str(offset),
            },
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        query = payload.get("query", {})
        search_results = query.get("search", []) if isinstance(query, Mapping) else []
        if not isinstance(search_results, list):
            raise WikidataBioRubricSourceError(
                "Wikidata search response did not contain a search list."
            )
        if not search_results:
            break

        page_entity_ids: list[str] = []
        for result in search_results:
            if not isinstance(result, Mapping):
                raise WikidataBioRubricSourceError(
                    "Wikidata search result entries must be objects."
                )
            title = result.get("title")
            if isinstance(title, str) and title.startswith("Q"):
                page_entity_ids.append(title)
        entity_ids.extend(page_entity_ids)

        if len(search_results) < search_limit:
            break

        continuation = payload.get("continue", {})
        next_offset = continuation.get("sroffset") if isinstance(continuation, Mapping) else None
        if isinstance(next_offset, int):
            offset = next_offset
        else:
            offset += search_limit

        if page_index < max_pages - 1:
            time.sleep(sleep_seconds)

    return sorted(set(entity_ids))


def _sparql_entity_id_query(slot: str, *, limit: int, offset: int) -> str:
    property_id = BIORUBRIC_SLOT_PROPERTY_IDS[slot]
    return f"""
SELECT DISTINCT ?entity WHERE {{
  ?entity wdt:P31 wd:Q5 .
  ?entity wdt:P106 ?occupation .
  ?entity wdt:{property_id} ?value .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def _fetch_entity_ids_via_sparql(
    slot: str,
    *,
    page_size: int,
    max_pages: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[str]:
    if slot not in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}:
        raise WikidataBioRubricSourceError("SPARQL fetches must target one distinguishing slot.")
    if page_size <= 0 or max_pages <= 0:
        raise WikidataBioRubricSourceError("page_size and max_pages must be positive integers.")

    entity_ids: list[str] = []
    for page_index in range(max_pages):
        payload = _request_sparql_json(
            _sparql_entity_id_query(slot, limit=page_size, offset=page_index * page_size),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        results = payload.get("results", {})
        bindings = results.get("bindings", []) if isinstance(results, Mapping) else []
        if not isinstance(bindings, list):
            raise WikidataBioRubricSourceError(
                "Wikidata SPARQL response did not contain a bindings list."
            )
        if not bindings:
            break

        page_entity_ids: list[str] = []
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise WikidataBioRubricSourceError("Wikidata SPARQL bindings must be objects.")
            entity_payload = binding.get("entity")
            if not isinstance(entity_payload, Mapping):
                continue
            entity_value = entity_payload.get("value")
            if isinstance(entity_value, str):
                page_entity_ids.append(_extract_entity_id(entity_value))
        entity_ids.extend(page_entity_ids)

        if len(bindings) < page_size:
            break
        if page_index < max_pages - 1:
            time.sleep(sleep_seconds)

    return sorted(set(entity_ids))


def _sparql_slot_rows_for_entities_query(slot: str, entity_ids: list[str]) -> str:
    property_id = BIORUBRIC_SLOT_PROPERTY_IDS[slot]
    values_clause = " ".join(f"wd:{entity_id}" for entity_id in entity_ids)
    return f"""
SELECT DISTINCT ?entity ?entityLabel ?value ?valueLabel WHERE {{
  VALUES ?entity {{ {values_clause} }}
  ?entity wdt:P31 wd:Q5 .
  ?entity wdt:{property_id} ?value .
  ?entity rdfs:label ?entityLabel .
  FILTER(LANG(?entityLabel) = "en")
  ?value rdfs:label ?valueLabel .
  FILTER(LANG(?valueLabel) = "en")
}}
""".strip()


def _fetch_slot_rows_for_entities_via_sparql(
    slot: str,
    entity_ids: Iterable[str],
    *,
    batch_size: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    deduped_entity_ids = sorted(set(entity_ids))
    if not deduped_entity_ids:
        return []
    if batch_size <= 0:
        raise WikidataBioRubricSourceError("batch_size must be a positive integer.")

    property_id = BIORUBRIC_SLOT_PROPERTY_IDS[slot]
    slot_rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(deduped_entity_ids), batch_size):
        batch_ids = deduped_entity_ids[batch_start : batch_start + batch_size]
        payload = _request_sparql_json(
            _sparql_slot_rows_for_entities_query(slot, batch_ids),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        results = payload.get("results", {})
        bindings = results.get("bindings", []) if isinstance(results, Mapping) else []
        if not isinstance(bindings, list):
            raise WikidataBioRubricSourceError(
                "Wikidata SPARQL response did not contain a bindings list."
            )

        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise WikidataBioRubricSourceError("Wikidata SPARQL bindings must be objects.")
            entity_payload = binding.get("entity")
            entity_label_payload = binding.get("entityLabel")
            value_payload = binding.get("value")
            value_label_payload = binding.get("valueLabel")
            if not all(
                isinstance(payload, Mapping)
                for payload in (
                    entity_payload,
                    entity_label_payload,
                    value_payload,
                    value_label_payload,
                )
            ):
                continue

            entity_value = entity_payload.get("value")
            entity_name = entity_label_payload.get("value")
            value_value = value_payload.get("value")
            value_text = value_label_payload.get("value")
            if not all(
                isinstance(value, str)
                for value in (entity_value, entity_name, value_value, value_text)
            ):
                continue

            entity_id = _extract_entity_id(entity_value)
            value_id = _extract_entity_id(value_value)
            slot_rows.append(
                {
                    "entity_id": entity_id,
                    "entity_name": entity_name.strip(),
                    "is_human": True,
                    "slot": slot,
                    "value_id": value_id,
                    "value_text": value_text.strip(),
                    "source_ref": f"wikidata:{entity_id}:{property_id}:{value_id}",
                }
            )

        if batch_start + batch_size < len(deduped_entity_ids):
            time.sleep(sleep_seconds)

    deduped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in slot_rows:
        deduped_rows[(row["entity_id"], row["slot"], row["value_id"])] = row
    return [deduped_rows[key] for key in sorted(deduped_rows)]


def _fetch_entity_payloads(
    entity_ids: Iterable[str],
    *,
    include_claims: bool,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> dict[str, dict[str, Any]]:
    deduped_entity_ids = sorted(set(entity_ids))
    if not deduped_entity_ids:
        return {}

    props = "labels|claims" if include_claims else "labels"
    payloads: dict[str, dict[str, Any]] = {}
    for batch_start in range(0, len(deduped_entity_ids), WIKIDATA_MAX_ENTITY_IDS_PER_REQUEST):
        batch_ids = deduped_entity_ids[
            batch_start : batch_start + WIKIDATA_MAX_ENTITY_IDS_PER_REQUEST
        ]
        payload = _request_api_json(
            {
                "action": "wbgetentities",
                "ids": "|".join(batch_ids),
                "props": props,
                "languages": "en",
            },
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        entities = payload.get("entities")
        if not isinstance(entities, Mapping):
            raise WikidataBioRubricSourceError(
                "wbgetentities response did not contain an entities mapping."
            )
        for entity_id in batch_ids:
            entity_payload = entities.get(entity_id)
            if not isinstance(entity_payload, Mapping):
                continue
            if entity_payload.get("missing") == "":
                continue
            payloads[entity_id] = dict(entity_payload)
        if batch_start + WIKIDATA_MAX_ENTITY_IDS_PER_REQUEST < len(deduped_entity_ids):
            time.sleep(sleep_seconds)
    return payloads


def _rows_from_entity_payload(
    entity_id: str,
    entity_payload: Mapping[str, Any],
    *,
    value_labels: Mapping[str, str],
) -> tuple[list[dict[str, Any]], set[str]]:
    claims = entity_payload.get("claims")
    if not isinstance(claims, Mapping):
        return [], set()

    if "Q5" not in _extract_truthy_entity_ids_from_claims(claims, "P31"):
        return [], set()

    entity_name = _label_from_entity_payload(entity_payload)
    if entity_name is None:
        return [], set()

    rows: list[dict[str, Any]] = []
    eligible_slots: set[str] = set()

    occupation_ids = _extract_truthy_entity_ids_from_claims(claims, "P106")
    for occupation_id in occupation_ids:
        occupation_label = value_labels.get(occupation_id)
        if occupation_label is None:
            continue
        rows.append(
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "is_human": True,
                "slot": "occupation",
                "value_id": occupation_id,
                "value_text": occupation_label,
                "source_ref": f"wikidata:{entity_id}:P106:{occupation_id}",
            }
        )
    if not rows:
        return [], set()

    slot_rows: list[dict[str, Any]] = []
    for slot in sorted(BIORUBRIC_ALLOWED_SLOTS - {"occupation"}):
        property_id = BIORUBRIC_SLOT_PROPERTY_IDS[slot]
        value_ids = _extract_truthy_entity_ids_from_claims(claims, property_id)
        slot_value_rows = [
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "is_human": True,
                "slot": slot,
                "value_id": value_id,
                "value_text": value_labels[value_id],
                "source_ref": f"wikidata:{entity_id}:{property_id}:{value_id}",
            }
            for value_id in value_ids
            if value_id in value_labels
        ]
        if len(slot_value_rows) >= 2:
            eligible_slots.add(slot)
        slot_rows.extend(slot_value_rows)

    if not eligible_slots:
        return [], set()

    rows.extend(slot_rows)
    deduped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        deduped_rows[(row["entity_id"], row["slot"], row["value_id"])] = row
    return [deduped_rows[key] for key in sorted(deduped_rows)], eligible_slots


def fetch_biorubric_source_rows_from_config(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch raw BioRubric truthy rows according to a JSON-compatible config."""

    slot_requests = config.get("slot_requests")
    if not isinstance(slot_requests, list) or not slot_requests:
        raise WikidataBioRubricSourceError("Config must contain a non-empty slot_requests list.")

    timeout_seconds = int(config.get("timeout_seconds", 60))
    max_retries = int(config.get("max_retries", 3))
    sleep_seconds = float(config.get("sleep_seconds", 1.0))
    retrieval_mode = config.get("retrieval_mode", "search_api")
    if retrieval_mode not in {"search_api", "sparql"}:
        raise WikidataBioRubricSourceError(
            "retrieval_mode must be either 'search_api' or 'sparql' when provided."
        )
    sparql_batch_size = int(config.get("sparql_batch_size", 100))
    if sparql_batch_size <= 0:
        raise WikidataBioRubricSourceError("sparql_batch_size must be a positive integer.")

    slot_request_summary: list[dict[str, Any]] = []
    requested_entity_ids_by_slot: dict[str, list[str]] = {}
    all_requested_entity_ids: set[str] = set()
    if retrieval_mode == "sparql":
        for request in slot_requests:
            if not isinstance(request, Mapping):
                raise WikidataBioRubricSourceError("Each slot_requests entry must be an object.")
            slot = request.get("slot")
            page_size = request.get("page_size", 50)
            max_pages = request.get("max_pages", 1)
            if not isinstance(slot, str):
                raise WikidataBioRubricSourceError("Each slot request must contain a string slot.")
            if not isinstance(page_size, int) or not isinstance(max_pages, int):
                raise WikidataBioRubricSourceError(
                    "Each slot request must contain integer page_size and max_pages."
                )
            if slot not in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}:
                raise WikidataBioRubricSourceError(
                    f"Slot request {slot!r} is not a supported distinguishing slot."
                )

            entity_ids = _fetch_entity_ids_via_sparql(
                slot,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            requested_entity_ids_by_slot[slot] = entity_ids
            all_requested_entity_ids.update(entity_ids)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "biorubric_entity_ids_fetched",
                        "slot": slot,
                        "entity_count": len(entity_ids),
                    }
                )
            slot_request_summary.append(
                {
                    "slot": slot,
                    "property_id": BIORUBRIC_SLOT_PROPERTY_IDS[slot],
                    "retrieval_mode": retrieval_mode,
                    "page_size_requested": page_size,
                    "page_size_effective": page_size,
                    "max_pages": max_pages,
                    "search_result_entity_count": len(entity_ids),
                }
            )

        entity_ids = sorted(all_requested_entity_ids)
        rows_by_slot: dict[str, list[dict[str, Any]]] = {}
        for slot in sorted(BIORUBRIC_ALLOWED_SLOTS):
            slot_rows = _fetch_slot_rows_for_entities_via_sparql(
                slot,
                entity_ids,
                batch_size=sparql_batch_size,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            rows_by_slot[slot] = slot_rows
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "biorubric_slot_rows_fetched",
                        "slot": slot,
                        "row_count": len(slot_rows),
                    }
                )

        rows_by_entity: dict[str, list[dict[str, Any]]] = {}
        for slot_rows in rows_by_slot.values():
            for row in slot_rows:
                rows_by_entity.setdefault(str(row["entity_id"]), []).append(row)

        final_rows: list[dict[str, Any]] = []
        eligible_entities_by_slot: dict[str, set[str]] = {
            slot: set() for slot in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}
        }
        for entity_id in sorted(rows_by_entity):
            entity_rows = rows_by_entity[entity_id]
            occupation_rows = [row for row in entity_rows if row["slot"] == "occupation"]
            if not occupation_rows:
                continue

            eligible_slots = {
                slot
                for slot in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}
                if len([row for row in entity_rows if row["slot"] == slot]) >= 2
            }
            if not eligible_slots:
                continue

            final_rows.extend(entity_rows)
            for slot in eligible_slots:
                eligible_entities_by_slot[slot].add(entity_id)

        deduped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in final_rows:
            deduped_rows[(row["entity_id"], row["slot"], row["value_id"])] = row
        final_rows = [deduped_rows[key] for key in sorted(deduped_rows)]
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "biorubric_rows_built",
                    "row_count": len(final_rows),
                    "entity_count": len({row["entity_id"] for row in final_rows}),
                }
            )

        slot_request_summary = [
            {
                **slot_summary,
                "eligible_entity_count": len(eligible_entities_by_slot[slot_summary["slot"]]),
                "row_count": len(
                    [row for row in final_rows if row["slot"] == slot_summary["slot"]]
                ),
            }
            for slot_summary in slot_request_summary
        ]
        summary = {
            "stage": "raw_biorubric_wikidata_fetch",
            "row_count": len(final_rows),
            "entity_count": len({row["entity_id"] for row in final_rows}),
            "slot_row_counts": dict(sorted(Counter(row["slot"] for row in final_rows).items())),
            "slot_entity_counts": {
                slot: len({row["entity_id"] for row in final_rows if row["slot"] == slot})
                for slot in sorted(BIORUBRIC_ALLOWED_SLOTS)
            },
            "slot_requests": slot_request_summary,
        }
        return final_rows, summary

    for request in slot_requests:
        if not isinstance(request, Mapping):
            raise WikidataBioRubricSourceError("Each slot_requests entry must be an object.")
        slot = request.get("slot")
        page_size = request.get("page_size", 50)
        max_pages = request.get("max_pages", 1)
        if not isinstance(slot, str):
            raise WikidataBioRubricSourceError("Each slot request must contain a string slot.")
        if not isinstance(page_size, int) or not isinstance(max_pages, int):
            raise WikidataBioRubricSourceError(
                "Each slot request must contain integer page_size and max_pages."
            )
        if slot not in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}:
            raise WikidataBioRubricSourceError(
                f"Slot request {slot!r} is not a supported distinguishing slot."
            )

        if retrieval_mode == "sparql":
            entity_ids = _fetch_entity_ids_via_sparql(
                slot,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        else:
            entity_ids = _fetch_search_entity_ids(
                slot,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        requested_entity_ids_by_slot[slot] = entity_ids
        all_requested_entity_ids.update(entity_ids)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "biorubric_entity_ids_fetched",
                    "slot": slot,
                    "entity_count": len(entity_ids),
                }
            )
        slot_request_summary.append(
            {
                "slot": slot,
                "property_id": BIORUBRIC_SLOT_PROPERTY_IDS[slot],
                "retrieval_mode": retrieval_mode,
                "page_size_requested": page_size,
                "page_size_effective": (
                    page_size
                    if retrieval_mode == "sparql"
                    else min(page_size, WIKIDATA_MAX_SEARCH_RESULTS)
                ),
                "max_pages": max_pages,
                "search_result_entity_count": len(entity_ids),
            }
        )

    entity_payloads = _fetch_entity_payloads(
        all_requested_entity_ids,
        include_claims=True,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "biorubric_entity_payloads_fetched",
                "entity_count": len(entity_payloads),
            }
        )

    value_ids: set[str] = set()
    for entity_payload in entity_payloads.values():
        claims = entity_payload.get("claims")
        if not isinstance(claims, Mapping):
            continue
        for property_id in ("P106", "P800", "P166", "P101"):
            value_ids.update(_extract_truthy_entity_ids_from_claims(claims, property_id))

    value_payloads = _fetch_entity_payloads(
        value_ids,
        include_claims=False,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "biorubric_value_payloads_fetched",
                "value_count": len(value_payloads),
            }
        )
    value_labels = {
        value_id: label
        for value_id, payload in value_payloads.items()
        if (label := _label_from_entity_payload(payload)) is not None
    }

    all_rows: list[dict[str, Any]] = []
    eligible_entities_by_slot: dict[str, set[str]] = {
        slot: set() for slot in BIORUBRIC_ALLOWED_SLOTS - {"occupation"}
    }
    for entity_id in sorted(entity_payloads):
        rows, eligible_slots = _rows_from_entity_payload(
            entity_id,
            entity_payloads[entity_id],
            value_labels=value_labels,
        )
        all_rows.extend(rows)
        for slot in eligible_slots:
            eligible_entities_by_slot[slot].add(entity_id)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "biorubric_rows_built",
                "row_count": len(all_rows),
                "entity_count": len(entity_payloads),
            }
        )

    deduped_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in all_rows:
        deduped_rows[(row["entity_id"], row["slot"], row["value_id"])] = row
    final_rows = [deduped_rows[key] for key in sorted(deduped_rows)]

    slot_request_summary = [
        {
            **slot_summary,
            "eligible_entity_count": len(eligible_entities_by_slot[slot_summary["slot"]]),
            "row_count": len([row for row in final_rows if row["slot"] == slot_summary["slot"]]),
        }
        for slot_summary in slot_request_summary
    ]

    summary = {
        "stage": "raw_biorubric_wikidata_fetch",
        "row_count": len(final_rows),
        "entity_count": len({row["entity_id"] for row in final_rows}),
        "slot_row_counts": dict(sorted(Counter(row["slot"] for row in final_rows).items())),
        "slot_entity_counts": {
            slot: len({row["entity_id"] for row in final_rows if row["slot"] == slot})
            for slot in sorted(BIORUBRIC_ALLOWED_SLOTS)
        },
        "slot_requests": slot_request_summary,
    }
    return final_rows, summary


def write_biorubric_source_artifacts(
    output_path: str | Path,
    summary_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True) + "\n")

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")

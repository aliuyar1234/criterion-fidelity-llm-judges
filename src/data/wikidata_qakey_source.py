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

from src.data.constants import QAKEY_RELATION_SPECS

WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_USER_AGENT = "criteria-over-beliefs-build/0.1 (research automation)"
WIKIDATA_MAX_SEARCH_RESULTS = 50
WIKIDATA_MAX_ENTITY_IDS_PER_REQUEST = 50
QAKEY_ENTITY_VALUED_RELATIONS = {"P19", "P20", "P36", "P38"}
QAKEY_TIME_VALUED_RELATIONS = {"P569", "P570"}


class WikidataQAKeySourceError(ValueError):
    """Raised when Wikidata QA-Key source extraction fails."""


def _extract_entity_id(entity_url: str) -> str:
    prefix = "http://www.wikidata.org/entity/"
    secure_prefix = "https://www.wikidata.org/entity/"
    if entity_url.startswith(prefix):
        return entity_url[len(prefix) :]
    if entity_url.startswith(secure_prefix):
        return entity_url[len(secure_prefix) :]
    raise WikidataQAKeySourceError(f"Expected a Wikidata entity URL, got {entity_url!r}.")


def _search_query_for_relation(relation_id: str, subject_class_qid: str) -> str:
    return f"haswbstatement:P31={subject_class_qid} haswbstatement:{relation_id}"


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

    raise WikidataQAKeySourceError(f"Wikidata request failed after retries: {last_error}")


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


def _claim_entity_target_id(claim: Mapping[str, Any]) -> str | None:
    mainsnak = claim.get("mainsnak")
    if not isinstance(mainsnak, Mapping) or mainsnak.get("snaktype") != "value":
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


def _format_wikidata_time_value(value: Mapping[str, Any]) -> str | None:
    time_value = value.get("time")
    precision = value.get("precision")
    if not isinstance(time_value, str) or not isinstance(precision, int):
        return None
    if len(time_value) < 11:
        return None

    sign = "-" if time_value.startswith("-") else ""
    stripped = time_value.lstrip("+-")
    year = stripped[0:4]
    month = stripped[5:7]
    day = stripped[8:10]

    if precision >= 11:
        return f"{sign}{year}-{month}-{day}"
    if precision == 10:
        return f"{sign}{year}-{month}"
    if precision == 9:
        return f"{sign}{year}"
    return None


def _extract_truthy_claim_values(
    claims: Mapping[str, Any],
    property_id: str,
) -> list[dict[str, Any]]:
    claim_list = claims.get(property_id, [])
    if not isinstance(claim_list, list):
        return []

    preferred_values: list[dict[str, Any]] = []
    normal_values: list[dict[str, Any]] = []
    for claim in claim_list:
        if not isinstance(claim, Mapping):
            continue
        rank = claim.get("rank")
        if rank == "deprecated":
            continue

        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, Mapping) or mainsnak.get("snaktype") != "value":
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, Mapping):
            continue
        value = datavalue.get("value")

        payload = {"rank": rank, "value": value}
        if rank == "preferred":
            preferred_values.append(payload)
        elif rank == "normal":
            normal_values.append(payload)

    return preferred_values if preferred_values else normal_values


def _fetch_search_entity_ids(
    relation_id: str,
    *,
    subject_class_qid: str,
    page_size: int,
    max_pages: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[str]:
    if relation_id not in QAKEY_RELATION_SPECS:
        raise WikidataQAKeySourceError(f"Relation {relation_id!r} is not in the frozen whitelist.")
    if page_size <= 0 or max_pages <= 0:
        raise WikidataQAKeySourceError("page_size and max_pages must be positive integers.")

    search_limit = min(page_size, WIKIDATA_MAX_SEARCH_RESULTS)
    offset = 0
    entity_ids: list[str] = []
    for page_index in range(max_pages):
        payload = _request_api_json(
            {
                "action": "query",
                "list": "search",
                "srsearch": _search_query_for_relation(relation_id, subject_class_qid),
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
            raise WikidataQAKeySourceError(
                "Wikidata search response did not contain a search list."
            )
        if not search_results:
            break

        page_entity_ids: list[str] = []
        for result in search_results:
            if not isinstance(result, Mapping):
                raise WikidataQAKeySourceError("Wikidata search results must be objects.")
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


def _sparql_subject_id_query(
    relation_id: str,
    *,
    subject_class_qid: str,
    limit: int,
    offset: int,
) -> str:
    return f"""
SELECT DISTINCT ?subject WHERE {{
  ?subject wdt:P31 wd:{subject_class_qid} .
  ?subject wdt:{relation_id} ?object .
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def _fetch_subject_entity_ids_via_sparql(
    relation_id: str,
    *,
    subject_class_qid: str,
    page_size: int,
    max_pages: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[str]:
    if relation_id not in QAKEY_RELATION_SPECS:
        raise WikidataQAKeySourceError(f"Relation {relation_id!r} is not in the frozen whitelist.")
    if page_size <= 0 or max_pages <= 0:
        raise WikidataQAKeySourceError("page_size and max_pages must be positive integers.")

    entity_ids: list[str] = []
    for page_index in range(max_pages):
        payload = _request_sparql_json(
            _sparql_subject_id_query(
                relation_id,
                subject_class_qid=subject_class_qid,
                limit=page_size,
                offset=page_index * page_size,
            ),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        results = payload.get("results", {})
        bindings = results.get("bindings", []) if isinstance(results, Mapping) else []
        if not isinstance(bindings, list):
            raise WikidataQAKeySourceError(
                "Wikidata SPARQL response did not contain a bindings list."
            )
        if not bindings:
            break

        page_entity_ids: list[str] = []
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise WikidataQAKeySourceError("Wikidata SPARQL bindings must be objects.")
            subject_payload = binding.get("subject")
            if not isinstance(subject_payload, Mapping):
                continue
            subject_value = subject_payload.get("value")
            if isinstance(subject_value, str):
                page_entity_ids.append(_extract_entity_id(subject_value))
        entity_ids.extend(page_entity_ids)

        if len(bindings) < page_size:
            break
        if page_index < max_pages - 1:
            time.sleep(sleep_seconds)

    return sorted(set(entity_ids))


def _sparql_entity_relation_rows_query(
    relation_id: str,
    *,
    subject_class_qid: str,
    limit: int,
    offset: int,
) -> str:
    return f"""
SELECT DISTINCT ?subject ?subjectLabel ?object ?objectLabel WHERE {{
  ?subject wdt:P31 wd:{subject_class_qid} .
  ?subject wdt:{relation_id} ?object .
  ?subject rdfs:label ?subjectLabel .
  FILTER(LANG(?subjectLabel) = "en")
  ?object rdfs:label ?objectLabel .
  FILTER(LANG(?objectLabel) = "en")
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def _sparql_date_relation_rows_query(
    relation_id: str,
    *,
    subject_class_qid: str,
    limit: int,
    offset: int,
) -> str:
    return f"""
SELECT DISTINCT ?subject ?subjectLabel ?time ?precision WHERE {{
  ?subject wdt:P31 wd:{subject_class_qid} .
  ?subject p:{relation_id} ?statement .
  ?statement wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  ?statement psv:{relation_id} ?valueNode .
  ?valueNode wikibase:timeValue ?time ;
             wikibase:timePrecision ?precision .
  ?subject rdfs:label ?subjectLabel .
  FILTER(LANG(?subjectLabel) = "en")
  FILTER(
    ?rank = wikibase:PreferredRank ||
    NOT EXISTS {{
      ?subject p:{relation_id} ?preferredStatement .
      ?preferredStatement wikibase:rank wikibase:PreferredRank .
    }}
  )
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def _fetch_relation_rows_via_sparql(
    relation_id: str,
    *,
    subject_class_qid: str,
    page_size: int,
    max_pages: int,
    timeout_seconds: int,
    max_retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    if relation_id not in QAKEY_RELATION_SPECS:
        raise WikidataQAKeySourceError(f"Relation {relation_id!r} is not in the frozen whitelist.")
    if page_size <= 0 or max_pages <= 0:
        raise WikidataQAKeySourceError("page_size and max_pages must be positive integers.")

    relation_rows: list[dict[str, Any]] = []
    for page_index in range(max_pages):
        if relation_id in QAKEY_ENTITY_VALUED_RELATIONS:
            query = _sparql_entity_relation_rows_query(
                relation_id,
                subject_class_qid=subject_class_qid,
                limit=page_size,
                offset=page_index * page_size,
            )
        else:
            query = _sparql_date_relation_rows_query(
                relation_id,
                subject_class_qid=subject_class_qid,
                limit=page_size,
                offset=page_index * page_size,
            )

        payload = _request_sparql_json(
            query,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        results = payload.get("results", {})
        bindings = results.get("bindings", []) if isinstance(results, Mapping) else []
        if not isinstance(bindings, list):
            raise WikidataQAKeySourceError(
                "Wikidata SPARQL response did not contain a bindings list."
            )
        if not bindings:
            break

        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise WikidataQAKeySourceError("Wikidata SPARQL bindings must be objects.")

            subject_payload = binding.get("subject")
            subject_label_payload = binding.get("subjectLabel")
            if not isinstance(subject_payload, Mapping) or not isinstance(
                subject_label_payload, Mapping
            ):
                continue
            subject_value = subject_payload.get("value")
            subject_text = subject_label_payload.get("value")
            if not isinstance(subject_value, str) or not isinstance(subject_text, str):
                continue

            subject_id = _extract_entity_id(subject_value)
            if relation_id in QAKEY_ENTITY_VALUED_RELATIONS:
                object_payload = binding.get("object")
                object_label_payload = binding.get("objectLabel")
                if not isinstance(object_payload, Mapping) or not isinstance(
                    object_label_payload, Mapping
                ):
                    continue
                object_value = object_payload.get("value")
                object_text = object_label_payload.get("value")
                if not isinstance(object_value, str) or not isinstance(object_text, str):
                    continue
                object_id = _extract_entity_id(object_value)
                relation_rows.append(
                    {
                        "subject_id": subject_id,
                        "subject_text": subject_text.strip(),
                        "relation_id": relation_id,
                        "object_id": object_id,
                        "object_text": object_text.strip(),
                        "object_aliases": [],
                        "source_ref": f"wikidata:{subject_id}:{relation_id}:{object_id}",
                    }
                )
            else:
                time_payload = binding.get("time")
                precision_payload = binding.get("precision")
                if not isinstance(time_payload, Mapping) or not isinstance(
                    precision_payload, Mapping
                ):
                    continue
                time_value = time_payload.get("value")
                precision_value = precision_payload.get("value")
                if not isinstance(time_value, str) or not isinstance(precision_value, str):
                    continue
                object_text = _format_wikidata_time_value(
                    {"time": time_value, "precision": int(precision_value)}
                )
                if object_text is None:
                    continue
                relation_rows.append(
                    {
                        "subject_id": subject_id,
                        "subject_text": subject_text.strip(),
                        "relation_id": relation_id,
                        "object_id": None,
                        "object_text": object_text,
                        "object_aliases": [],
                        "source_ref": f"wikidata:{subject_id}:{relation_id}:{object_text}",
                    }
                )

        if len(bindings) < page_size:
            break
        if page_index < max_pages - 1:
            time.sleep(sleep_seconds)

    deduped_relation_rows: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
    for row in relation_rows:
        deduped_relation_rows[
            (row["subject_id"], row["relation_id"], row["object_id"], row["object_text"])
        ] = row
    return [deduped_relation_rows[key] for key in sorted(deduped_relation_rows)]


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
            raise WikidataQAKeySourceError(
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


def _rows_from_subject_payload(
    subject_id: str,
    relation_id: str,
    subject_payload: Mapping[str, Any],
    *,
    object_labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    subject_text = _label_from_entity_payload(subject_payload)
    claims = subject_payload.get("claims")
    if subject_text is None or not isinstance(claims, Mapping):
        return []

    rows: list[dict[str, Any]] = []
    property_values = _extract_truthy_claim_values(claims, relation_id)
    for property_value in property_values:
        value = property_value["value"]
        if relation_id in QAKEY_ENTITY_VALUED_RELATIONS:
            if not isinstance(value, Mapping):
                continue
            object_id = value.get("id")
            if not isinstance(object_id, str) or object_id not in object_labels:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "subject_text": subject_text,
                    "relation_id": relation_id,
                    "object_id": object_id,
                    "object_text": object_labels[object_id],
                    "object_aliases": [],
                    "source_ref": f"wikidata:{subject_id}:{relation_id}:{object_id}",
                }
            )
        elif relation_id in QAKEY_TIME_VALUED_RELATIONS:
            if not isinstance(value, Mapping):
                continue
            object_text = _format_wikidata_time_value(value)
            if object_text is None:
                continue
            rows.append(
                {
                    "subject_id": subject_id,
                    "subject_text": subject_text,
                    "relation_id": relation_id,
                    "object_id": None,
                    "object_text": object_text,
                    "object_aliases": [],
                    "source_ref": f"wikidata:{subject_id}:{relation_id}:{object_text}",
                }
            )
        else:  # pragma: no cover - whitelist enforcement should keep this unreachable
            raise WikidataQAKeySourceError(f"Unsupported QA-Key relation {relation_id!r}.")

    deduped_rows: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    for row in rows:
        deduped_rows[
            (row["subject_id"], row["relation_id"], row["object_id"], row["object_text"])
        ] = row
    return [deduped_rows[key] for key in sorted(deduped_rows)]


def fetch_qakey_source_rows_from_config(
    config: Mapping[str, Any],
    *,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch raw QA-Key source rows according to a JSON-compatible config."""

    relation_requests = config.get("relation_requests")
    if not isinstance(relation_requests, list) or not relation_requests:
        raise WikidataQAKeySourceError("Config must contain a non-empty relation_requests list.")

    timeout_seconds = int(config.get("timeout_seconds", 60))
    max_retries = int(config.get("max_retries", 3))
    sleep_seconds = float(config.get("sleep_seconds", 1.0))
    retrieval_mode = config.get("retrieval_mode", "search_api")
    if retrieval_mode not in {"search_api", "sparql"}:
        raise WikidataQAKeySourceError(
            "retrieval_mode must be either 'search_api' or 'sparql' when provided."
        )

    requested_subject_ids_by_relation: dict[str, list[str]] = {}
    relation_config_summary: list[dict[str, Any]] = []
    all_subject_ids: set[str] = set()
    if retrieval_mode == "sparql":
        all_rows: list[dict[str, Any]] = []
        relation_row_counts: Counter[str] = Counter()
        for request in relation_requests:
            if not isinstance(request, Mapping):
                raise WikidataQAKeySourceError("Each relation_requests entry must be an object.")
            relation_id = request.get("relation_id")
            subject_class_qid = request.get("subject_class_qid")
            page_size = request.get("page_size", 50)
            max_pages = request.get("max_pages", 1)
            if not isinstance(relation_id, str) or not isinstance(subject_class_qid, str):
                raise WikidataQAKeySourceError(
                    "Each relation request must contain string relation_id and subject_class_qid."
                )
            if not isinstance(page_size, int) or not isinstance(max_pages, int):
                raise WikidataQAKeySourceError(
                    "Each relation request must contain integer page_size and max_pages."
                )

            relation_rows = _fetch_relation_rows_via_sparql(
                relation_id,
                subject_class_qid=subject_class_qid,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
            relation_row_counts[relation_id] = len(relation_rows)
            all_rows.extend(relation_rows)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "qakey_relation_rows_built",
                        "relation_id": relation_id,
                        "row_count": len(relation_rows),
                    }
                )
            relation_config_summary.append(
                {
                    "relation_id": relation_id,
                    "subject_class_qid": subject_class_qid,
                    "retrieval_mode": retrieval_mode,
                    "page_size_requested": page_size,
                    "page_size_effective": page_size,
                    "max_pages": max_pages,
                    "search_result_subject_count": len(
                        {row["subject_id"] for row in relation_rows}
                    ),
                }
            )

        deduped_rows: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
        for row in all_rows:
            deduped_rows[
                (row["subject_id"], row["relation_id"], row["object_id"], row["object_text"])
            ] = row

        final_rows = [deduped_rows[key] for key in sorted(deduped_rows)]
        relation_config_summary = [
            {
                **relation_summary,
                "row_count": relation_row_counts[relation_summary["relation_id"]],
            }
            for relation_summary in relation_config_summary
        ]
        summary = {
            "stage": "raw_qakey_wikidata_fetch",
            "row_count": len(final_rows),
            "relation_row_counts": dict(sorted(relation_row_counts.items())),
            "relation_requests": relation_config_summary,
        }
        return final_rows, summary

    for request in relation_requests:
        if not isinstance(request, Mapping):
            raise WikidataQAKeySourceError("Each relation_requests entry must be an object.")
        relation_id = request.get("relation_id")
        subject_class_qid = request.get("subject_class_qid")
        page_size = request.get("page_size", 50)
        max_pages = request.get("max_pages", 1)
        if not isinstance(relation_id, str) or not isinstance(subject_class_qid, str):
            raise WikidataQAKeySourceError(
                "Each relation request must contain string relation_id and subject_class_qid."
            )
        if not isinstance(page_size, int) or not isinstance(max_pages, int):
            raise WikidataQAKeySourceError(
                "Each relation request must contain integer page_size and max_pages."
            )

        if retrieval_mode == "sparql":
            subject_ids = _fetch_subject_entity_ids_via_sparql(
                relation_id,
                subject_class_qid=subject_class_qid,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        else:
            subject_ids = _fetch_search_entity_ids(
                relation_id,
                subject_class_qid=subject_class_qid,
                page_size=page_size,
                max_pages=max_pages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep_seconds=sleep_seconds,
            )
        requested_subject_ids_by_relation[relation_id] = subject_ids
        all_subject_ids.update(subject_ids)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "qakey_subject_ids_fetched",
                    "relation_id": relation_id,
                    "subject_count": len(subject_ids),
                }
            )
        relation_config_summary.append(
            {
                "relation_id": relation_id,
                "subject_class_qid": subject_class_qid,
                "retrieval_mode": retrieval_mode,
                "page_size_requested": page_size,
                "page_size_effective": (
                    page_size
                    if retrieval_mode == "sparql"
                    else min(page_size, WIKIDATA_MAX_SEARCH_RESULTS)
                ),
                "max_pages": max_pages,
                "search_result_subject_count": len(subject_ids),
            }
        )

    subject_payloads = _fetch_entity_payloads(
        all_subject_ids,
        include_claims=True,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "qakey_subject_payloads_fetched",
                "subject_count": len(subject_payloads),
            }
        )

    object_ids: set[str] = set()
    for relation_id, subject_ids in requested_subject_ids_by_relation.items():
        if relation_id not in QAKEY_ENTITY_VALUED_RELATIONS:
            continue
        for subject_id in subject_ids:
            subject_payload = subject_payloads.get(subject_id)
            if subject_payload is None:
                continue
            claims = subject_payload.get("claims")
            if not isinstance(claims, Mapping):
                continue
            for property_value in _extract_truthy_claim_values(claims, relation_id):
                value = property_value["value"]
                if isinstance(value, Mapping):
                    object_id = value.get("id")
                    if isinstance(object_id, str):
                        object_ids.add(object_id)

    object_payloads = _fetch_entity_payloads(
        object_ids,
        include_claims=False,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "qakey_object_payloads_fetched",
                "object_count": len(object_payloads),
            }
        )
    object_labels = {
        object_id: label
        for object_id, payload in object_payloads.items()
        if (label := _label_from_entity_payload(payload)) is not None
    }

    all_rows: list[dict[str, Any]] = []
    relation_row_counts: Counter[str] = Counter()
    for relation_id, subject_ids in requested_subject_ids_by_relation.items():
        relation_rows: list[dict[str, Any]] = []
        for subject_id in subject_ids:
            subject_payload = subject_payloads.get(subject_id)
            if subject_payload is None:
                continue
            relation_rows.extend(
                _rows_from_subject_payload(
                    subject_id,
                    relation_id,
                    subject_payload,
                    object_labels=object_labels,
                )
            )
        deduped_relation_rows: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
        for row in relation_rows:
            deduped_relation_rows[
                (row["subject_id"], row["relation_id"], row["object_id"], row["object_text"])
            ] = row
        relation_rows = [deduped_relation_rows[key] for key in sorted(deduped_relation_rows)]
        relation_row_counts[relation_id] = len(relation_rows)
        all_rows.extend(relation_rows)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "qakey_relation_rows_built",
                    "relation_id": relation_id,
                    "row_count": len(relation_rows),
                }
            )

    deduped_rows: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
    for row in all_rows:
        deduped_rows[
            (row["subject_id"], row["relation_id"], row["object_id"], row["object_text"])
        ] = row

    final_rows = [deduped_rows[key] for key in sorted(deduped_rows)]
    relation_config_summary = [
        {
            **relation_summary,
            "row_count": relation_row_counts[relation_summary["relation_id"]],
        }
        for relation_summary in relation_config_summary
    ]
    summary = {
        "stage": "raw_qakey_wikidata_fetch",
        "row_count": len(final_rows),
        "relation_row_counts": dict(sorted(relation_row_counts.items())),
        "relation_requests": relation_config_summary,
    }
    return final_rows, summary


def write_qakey_source_artifacts(
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

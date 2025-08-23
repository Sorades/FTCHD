import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import asynccontextmanager

from common.logging import get_logger
from interoperability.fhir_bundle import decode_id

log = get_logger(__name__)


class FHIRBundleServer:
    def __init__(self, bundle_file_path: str):
        self.bundle_file_path = bundle_file_path
        self.resources = {}
        self.resource_types = {}
        self.load_bundle()

    def load_bundle(self):
        try:
            with open(self.bundle_file_path, "r", encoding="utf-8") as f:
                bundle_data = json.load(f)

            if bundle_data.get("resourceType") != "Bundle":
                raise ValueError("Not a valida FHIR Bundle")

            for entry in bundle_data.get("entry", []):
                resource = entry.get("resource")
                if not resource:
                    continue
                resource_type = resource.get("resourceType")
                resource_id = decode_id(resource.get("id"))
                if resource_id is not None:
                    resource["id"] = resource_id

                if resource_type and resource_id:
                    # 存储资源
                    key = f"{resource_type}/{resource_id}"
                    self.resources[key] = resource

                    # 按类型分组
                    if resource_type not in self.resource_types:
                        self.resource_types[resource_type] = {}
                    self.resource_types[resource_type][resource_id] = resource

            log.info(f"Load successfully: {len(self.resources)} FHIR resources")
            log.info(f"Resource types: {list(self.resource_types.keys())}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Bundle file not found: {self.bundle_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {self.bundle_file_path}")

    def get_resource_by_id(self, resource_type: str, resource_id: str) -> dict | None:
        key = f"{resource_type}/{resource_id}"
        return self.resources.get(key)

    def search_resources(
        self, resource_type: str, search_params: dict[str, Any]
    ) -> list[dict]:
        if resource_type not in self.resource_types:
            return []

        resources = list(self.resource_types[resource_type].values())

        # return all resources if search params is empty
        if not search_params:
            return resources

        # Apply search filters
        filtered_resources = []
        for resource in resources:
            if self._matches_search_criteria(resource, search_params):
                filtered_resources.append(resource)

        return filtered_resources

    def _matches_search_criteria(
        self, resource: dict, search_params: dict[str, Any]
    ) -> bool:
        """Check if the resource matches the search criteria"""
        for param_name, param_value in search_params.items():
            if not self._matches_parameter(resource, param_name, param_value):
                return False
        return True

    def _matches_parameter(
        self, resource: dict, param_name: str, param_value: str
    ) -> bool:
        """Check if the parameter matches the resource"""

        if param_name == "identifier":
            return self._search_in_identifiers(resource, param_value)

        if resource.get("resourceType") == "Patient":
            if param_name == "name":
                return self._search_in_names(resource, param_value)

        if resource.get("resourceType") == "DiagnosticReport":
            if param_name == "conclusion":
                conclusion = resource.get("conclusion", "")
                return param_value.lower() in conclusion.lower()
            if param_name == "status":
                return resource.get("status") == param_value

        if resource.get("resourceType") == "Organization":
            if param_name == "name":
                name = resource.get("name", "")
                return param_value.lower() in name.lower()

        if resource.get("resourceType") == "ImagingStudy":
            if param_name == "status":
                return resource.get("status") == param_value
            if param_name == "subject":
                subject_ref = resource.get("subject", {}).get("reference", "")
                return param_value in subject_ref

        if resource.get("resourceType") == "Observation":
            if param_name == "code":
                return self._search_in_code(resource.get("code"), param_value)  # type: ignore
            if param_name == "value-integer":
                return str(resource.get("valueInteger", "")) == param_value
            if param_name == "value-string":
                value_str = resource.get("valueString", "")
                return param_value.lower() in value_str.lower()

        return False

    def _search_in_identifiers(self, resource: dict, search_value: str) -> bool:
        identifiers = resource.get("identifier", [])
        for identifier in identifiers:
            if search_value in identifier.get("value", ""):
                return True
        return False

    def _search_in_names(self, resource: dict, search_value: str) -> bool:
        names = resource.get("name", [])
        for name in names:
            family = name.get("family", "")
            given = " ".join(name.get("given", []))
            full_name = f"{family} {given}".strip()
            if search_value.lower() in full_name.lower():
                return True
        return False

    def _search_in_code(self, code_concept: dict, search_value: str) -> bool:
        if not code_concept:
            return False
        codings = code_concept.get("coding", [])
        for coding in codings:
            if (
                search_value in coding.get("code", "")
                or search_value.lower() in coding.get("display", "").lower()
            ):
                return True
        return False


fhir_server: FHIRBundleServer | None = None


def create_app(bundle_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global fhir_server
        if bundle_path:
            try:
                fhir_server = FHIRBundleServer(bundle_path)
            except Exception as e:
                log.error(f"Failed to load Bundle file: {e}")
                fhir_server = None
        else:
            log.warning("File not found, server not initialized")

        yield
        fhir_server = None

    app = FastAPI(
        title="FHIR Query API Server",
        version="1.0.0",
    )

    @app.get("/")
    async def root():
        """Get server information"""
        return {
            "message": "FHIR Query API Server",
            "version": "1.0.0",
            "loaded_resources": len(fhir_server.resources) if fhir_server else 0,
            "resource_types": list(fhir_server.resource_types.keys())
            if fhir_server
            else [],
        }

    @app.get("/metadata")
    async def get_capability_statement():
        """Get server capability statement"""
        if not fhir_server:
            raise HTTPException(status_code=503, detail="Server not initialized")

        return {
            "resourceType": "CapabilityStatement",
            "status": "active",
            "date": datetime.now().isoformat(),
            "kind": "instance",
            "software": {"name": "FHIR Bundle Query Server", "version": "1.0.0"},
            "fhirVersion": "4.0.1",
            "format": ["json"],
            "rest": [
                {
                    "mode": "server",
                    "resource": [
                        {
                            "type": resource_type,
                            "interaction": [{"code": "read"}, {"code": "search-type"}],
                        }
                        for resource_type in fhir_server.resource_types.keys()
                    ],
                }
            ],
        }

    @app.get("/{resource_type}/{resource_id}")
    async def get_resource(resource_type: str, resource_id: str):
        if not fhir_server:
            raise HTTPException(status_code=503, detail="Server not initialized")

        resource = fhir_server.get_resource_by_id(resource_type, resource_id)
        if not resource:
            raise HTTPException(
                status_code=404,
                detail=f"Resources not found: {resource_type}/{resource_id}",
            )

        return resource

    @app.get("/{resource_type}")
    async def search_resources(
        resource_type: str,
        identifier: Optional[str] = Query(None, description="According identifier"),
        patient_id: Optional[str] = Query(
            None, description="Search according patient id"
        ),
        conclusion: Optional[str] = Query(
            None, description="Search according conclusion"
        ),
        status: Optional[str] = Query(None, description="Search according status"),
        org_name: Optional[str] = Query(
            None, alias="name", description="Search according organization name"
        ),
        subject: Optional[str] = Query(None, description="Search according subject"),
        code: Optional[str] = Query(
            None, description="Search according observation code"
        ),
        value_integer: Optional[str] = Query(
            None, alias="value-integer", description="Search according integer value"
        ),
        value_string: Optional[str] = Query(
            None, alias="value-string", description="Search according string value"
        ),
        _count: Optional[int] = Query(5, description="Return result count limit"),
        _offset: Optional[int] = Query(None, description="Result offset"),
    ):
        if not fhir_server:
            raise HTTPException(status_code=503, detail="Server not initialized")

        search_params = {}
        if identifier:
            search_params["identifier"] = identifier
        if patient_id:
            search_params["name"] = patient_id
        if conclusion:
            search_params["conclusion"] = conclusion
        if status:
            search_params["status"] = status
        if org_name:
            search_params["name"] = org_name
        if subject:
            search_params["subject"] = subject
        if code:
            search_params["code"] = code
        if value_integer:
            search_params["value-integer"] = value_integer
        if value_string:
            search_params["value-string"] = value_string

        resources = fhir_server.search_resources(resource_type, search_params)

        total = len(resources)
        if _offset:
            resources = resources[_offset:]
        if _count:
            resources = resources[:_count]

        search_bundle = {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": total,
            "entry": [
                {
                    "fullUrl": f"{resource.get('resourceType')}/{resource.get('id')}",
                    "resource": resource,
                }
                for resource in resources
            ],
        }

        return search_bundle

    return app


def main():
    parser = argparse.ArgumentParser(description="FHIR API")
    parser.add_argument("--task", choices=["binary", "subtype"], default="subtype")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    bundle_path = (
        Path("data/fhir/binary-data-bundle.json")
        if args.task == "binary"
        else Path("data/fhir/subtype-data-bundle.json")
    )
    if not bundle_path.exists():
        log.error(f"File is not found: {bundle_path}")
        return

    try:
        app = create_app(str(bundle_path))
        log.info(f"File loaded successfully: {bundle_path}")
    except Exception as e:
        log.error(f"Failed to load Bundle file: {e}")
        return

    log.info("Starting FHIR API server...")
    log.info(f"Address: http://{args.host}:{args.port}")
    log.info(f"API documentation: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


app = create_app()

if __name__ == "__main__":
    main()

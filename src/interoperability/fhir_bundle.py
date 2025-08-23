import argparse
import json
import uuid
from typing import Literal

import pandas as pd
from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.device import Device, DeviceName
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.imagingstudy import ImagingStudy, ImagingStudySeries
from fhir.resources.observation import Observation
from fhir.resources.organization import Organization
from fhir.resources.patient import Patient
from fhir.resources.reference import Reference
from tqdm.rich import tqdm

from common.logging import get_logger

SYS_SNOMED = "http://snomed.info/sct"
SYS_HOSPITAL = "https://zgcx.nhc.gov.cn/unit"
SYS_FTCHD = "https://github.com/Sorades/FTCHD/terminology"
SYS_HL7 = "http://terminology.hl7.org/CodeSystem/v2-0074"

HOSP_CODE_MAPPING = {
    "BOGH": "000028110101210611",
    "CH": "44490745543011111G1001",
    "GZAH": "49850124-X45010611G1001",
    "NWCH": "08818209861011311A5181",
    "PLAH": "PLA-General-Hospital",
    "TAHZU": "41580048041010111G1001",
    "XH": "44488501443010511A1001",
}
HOSP_FULLNAME_MAPPING = {
    "BOGH": "Beijing Obstetrics and Gynecology Hospital",
    "CH": "Changsha Hospital for Maternal and Child Health Care",
    "GZAH": "Maternal and Child Health Hospital of Guangxi Zhuang Autonomous Region",
    "NWCH": "Northwest Women’s and Children’s Hospital",
    "PLAH": "PLA General Hospital",
    "TAHZU": "the Third Affiliated Hospital of Zhengzhou University",
    "XH": "Xiangya Hospital of Central South University",
}

PREFIX = "ENC.-"

log = get_logger(__name__)


def encode_id(text: str):
    hex_str = text.encode("utf-8").hex()
    return PREFIX + hex_str.upper()


def decode_id(hex_text: str):
    if (
        hex_text is None
        or not isinstance(hex_text, str)
        or not hex_text.startswith(PREFIX)
    ):
        return hex_text

    decoded_bytes = bytes.fromhex(hex_text[len(PREFIX) :])
    return decoded_bytes.decode("utf-8")


class UltrasoundDataConverter:
    def convert_csv_to_fhir(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        fhir_bundle = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            patient = self.create_patient(row["CaseID"])

            organization = self.create_organization(row["Domain"])

            device = self.create_device(row["Device"])

            imaging_study = self.create_study(row["ID"], row["CaseID"])

            diagnostic_report = self.create_diagnosis(row["Label"], row["ID"])

            flow_observations = self.create_observations(
                row["CaseID"], row["ChamberNum"], row["FlowNum"], row["FlowSymmetry"]
            )

            fhir_bundle.extend(
                [patient, organization, device, imaging_study, diagnostic_report]
                + flow_observations
            )

        return fhir_bundle

    def create_patient(self, case_id):
        return Patient(
            id=case_id,
            identifier=[Identifier(system=SYS_FTCHD, use="official", value=case_id)],
        )

    def create_organization(self, domain):
        hosp = domain.replace("-pros", "").upper()

        code = HOSP_CODE_MAPPING[hosp]
        fullname = HOSP_FULLNAME_MAPPING[hosp]
        system = SYS_HOSPITAL if domain != "PLAH" else SYS_FTCHD
        return Organization(
            id=code,
            identifier=[Identifier(system=system, use="official", value=code)],
            name=fullname,
        )

    def create_device(self, device_name: str):
        return Device(
            name=[DeviceName(value=device_name, type="user-friendly-name")],
            type=[
                CodeableConcept(
                    coding=[
                        Coding(
                            system=SYS_SNOMED,
                            code="706332007",
                            display="Ultrasound imaging system",
                        )
                    ]
                )
            ],
        )

    def create_study(self, image_id: str, case_id: str):
        valid_id = encode_id(image_id)
        return ImagingStudy(
            id=valid_id,
            identifier=[Identifier(system=SYS_FTCHD, value=valid_id)],
            status="available",
            modality=[
                CodeableConcept(
                    coding=[
                        Coding(system=SYS_HL7, code="CUS", display="Cardiac Ultrasound")
                    ]
                )
            ],
            series=[
                ImagingStudySeries(
                    uid=str(uuid.uuid4()),
                    modality=CodeableConcept(
                        coding=[
                            Coding(
                                system=SYS_SNOMED,
                                code="711490008",
                                display="Doppler fetal echocardiography",
                            )
                        ]
                    ),
                )
            ],
            subject=Reference(reference=f"Patient/{case_id}"),
        )

    def create_diagnosis(self, label, image_id):
        return DiagnosticReport(
            id=encode_id(f"report-{image_id}"),
            status="final",
            category=[
                CodeableConcept(
                    coding=[
                        Coding(system=SYS_HL7, code="CUS", display="Cardiac Ultrasound")
                    ]
                )
            ],
            code=CodeableConcept(
                coding=[
                    Coding(
                        system=SYS_SNOMED,
                        code="711490008",
                        display="Doppler fetal echocardiography",
                    )
                ]
            ),
            subject=Reference(reference=f"ImagingStudy/{encode_id(image_id)}"),
            conclusion=label,
            conclusionCode=[
                CodeableConcept(
                    coding=[Coding(system=SYS_FTCHD, code=label, display=label)]
                )
            ],
        )

    def create_observations(
        self, study_id: str, chamber_num: int, flow_num: str, flow_symm: str
    ):
        obs_chamber_num = Observation(
            id=f"{study_id}-chamber-num",
            status="final",
            code=CodeableConcept(
                coding=[
                    Coding(
                        system=SYS_FTCHD,
                        code="chamber-num",
                        display="Chamber Number",
                    )
                ]
            ),
            subject=Reference(reference=f"ImagingStudy/{study_id}"),
            valueInteger=chamber_num,
        )
        obs_flow_num = Observation(
            id=f"{study_id}-flow-num",
            status="final",
            code=CodeableConcept(
                coding=[
                    Coding(
                        system=SYS_FTCHD,
                        code="flow-num",
                        display="Flow Number",
                    )
                ]
            ),
            subject=Reference(reference=f"ImagingStudy/{study_id}"),
            valueString=flow_num,
        )
        obs_flow_symm = Observation(
            id=f"{study_id}-flow-symm",
            status="final",
            code=CodeableConcept(
                coding=[
                    Coding(
                        system=SYS_FTCHD,
                        code="flow-symm",
                        display="Flow Symmetry",
                    )
                ]
            ),
            subject=Reference(reference=f"ImagingStudy/{study_id}"),
            valueString=flow_symm,
        )

        return [obs_chamber_num, obs_flow_num, obs_flow_symm]


def transfer(task: Literal["binary", "subtype"]):
    converter = UltrasoundDataConverter()

    csv_file = "data/binary_label.csv" if task == "binary" else "data/subtype_label.csv"
    fhir_resources = converter.convert_csv_to_fhir(csv_file)

    bundle = Bundle(id=f"{task}-data-bundle", type="collection")
    bundle.entry = []

    resource_counts = {}
    for resource in tqdm(fhir_resources):
        entry = BundleEntry()
        entry.resource = resource
        entry.fullUrl = f"urn:uuid:{resource.id}"
        bundle.entry.append(entry)

        resource_type = resource.__resource_type__
        resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1

    save_path = f"data/fhir/{task}-data-bundle.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(bundle.model_dump(), f, ensure_ascii=False, indent=2)

    log.info(
        f"Transfer finished. Generated {len(fhir_resources)} FHIR resources in {save_path}."
    )

    log.info("Stats:")
    log.info(resource_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "subtype"])
    args = parser.parse_args()

    transfer(args.task)

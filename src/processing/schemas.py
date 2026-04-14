"""
Pydantic schemas for processed window counts and dataset metadata.
"""
from typing import List
from pydantic import BaseModel, Field, NonNegativeInt


class WindowCount(BaseModel):
    tcp: NonNegativeInt = Field(..., description="TCP packet count in window")
    udp: NonNegativeInt = Field(..., description="UDP packet count in window")
    ssdp: NonNegativeInt = Field(..., description="SSDP (UDP:1900) packet count in window")
    arp: NonNegativeInt = Field(..., description="ARP packet count in window")


class Metadata(BaseModel):
    date: str
    file_source: str


class ProcessedDataset(BaseModel):
    metadata: Metadata
    windows: List[WindowCount]


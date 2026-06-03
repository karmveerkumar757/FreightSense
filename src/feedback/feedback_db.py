# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///data/freightsense.db"

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

Engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Engine)
Base = declarative_base()

class ShipmentRecord(Base):
    __tablename__ = "shipments"
    
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    raw_input_text = Column(Text, nullable=False)
    extracted_entities = Column(Text)  # JSON String of entities
    sentence_intents = Column(Text)    # JSON String of intents
    advisory_json = Column(Text)       # JSON String of full advisory
    overall_risk = Column(String)      # low, medium, high
    has_feedback = Column(Boolean, default=False)
    feedback_notes = Column(Text)      # Override notes
    corrected_data = Column(Text)      # Corrected JSON data if overridden

def init_db():
    """
    Creates the database tables if they do not exist.
    """
    Base.metadata.create_all(bind=Engine)

def get_db():
    """
    Yields database session local instances.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_shipment_to_db(shipment_id: str, raw_text: str, entities: list, intents: list, advisory: dict) -> ShipmentRecord:
    """
    Saves a shipment processing result to the SQLite DB.
    """
    import json
    db = SessionLocal()
    try:
        record = ShipmentRecord(
            id=shipment_id,
            timestamp=datetime.now(),
            raw_input_text=raw_text,
            extracted_entities=json.dumps(entities, ensure_ascii=False),
            sentence_intents=json.dumps(intents, ensure_ascii=False),
            advisory_json=json.dumps(advisory, ensure_ascii=False),
            overall_risk=advisory.get("overall_risk", "low")
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()

def update_shipment_feedback(shipment_id: str, feedback_notes: str, corrected_data: dict = None) -> bool:
    """
    Saves dispatcher overrides/feedback for MLOps retraining.
    """
    import json
    db = SessionLocal()
    try:
        record = db.query(ShipmentRecord).filter(ShipmentRecord.id == shipment_id).first()
        if record:
            record.has_feedback = True
            record.feedback_notes = feedback_notes
            if corrected_data:
                record.corrected_data = json.dumps(corrected_data, ensure_ascii=False)
            db.commit()
            return True
        return False
    finally:
        db.close()

def get_all_shipments() -> list:
    """
    Returns all historical shipment records.
    """
    db = SessionLocal()
    try:
        return db.query(ShipmentRecord).order_by(ShipmentRecord.timestamp.desc()).all()
    finally:
        db.close()

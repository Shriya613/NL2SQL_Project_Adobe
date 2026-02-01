BEGIN TRANSACTION;
CREATE TABLE Hospitals (
    HospitalID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Address TEXT
);
INSERT INTO "Hospitals" VALUES(1,'St. Mary''s General','123 Medical Avenue');
INSERT INTO "Hospitals" VALUES(2,'King''s College Hospital','456 Medical Road');
INSERT INTO "Hospitals" VALUES(3,'Queen Elizabeth II Hospital','789 Medical Street');
INSERT INTO "Hospitals" VALUES(4,'City General Hospital','1011 Medical Lane');
INSERT INTO "Hospitals" VALUES(5,'University Hospital','1213 Medical Circle');
CREATE TABLE Hospitals_Procedures (
    HospitalID INTEGER,
    ProcedureID INTEGER,
    FOREIGN KEY(HospitalID) REFERENCES Hospitals(HospitalID),
    FOREIGN KEY(ProcedureID) REFERENCES Procedures(ProcedureID),
    PRIMARY KEY(HospitalID, ProcedureID)
);
INSERT INTO "Hospitals_Procedures" VALUES(1,1);
INSERT INTO "Hospitals_Procedures" VALUES(2,2);
INSERT INTO "Hospitals_Procedures" VALUES(3,3);
INSERT INTO "Hospitals_Procedures" VALUES(4,4);
INSERT INTO "Hospitals_Procedures" VALUES(5,5);
INSERT INTO "Hospitals_Procedures" VALUES(2,1);
INSERT INTO "Hospitals_Procedures" VALUES(3,1);
INSERT INTO "Hospitals_Procedures" VALUES(4,1);
INSERT INTO "Hospitals_Procedures" VALUES(5,1);
CREATE TABLE Procedures (
    ProcedureID INTEGER PRIMARY KEY AUTOINCREMENT,
    ProcedureName TEXT,
    Description TEXT
);
INSERT INTO "Procedures" VALUES(1,'MRI Scan','Magnetic Resonance Imaging scan for diagnostic purposes.');
INSERT INTO "Procedures" VALUES(2,'CT Scan','Computed Tomography scan for diagnostic purposes.');
INSERT INTO "Procedures" VALUES(3,'Ultrasound','High-frequency sound waves used to visualize internal body structures.');
INSERT INTO "Procedures" VALUES(4,'X-ray','Use of ionizing radiation to produce images of the inside of a patient''s body.');
INSERT INTO "Procedures" VALUES(5,'Blood Test','Analysis of blood samples for various medical conditions.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hospitals',5);
INSERT INTO "sqlite_sequence" VALUES('Procedures',5);
COMMIT;

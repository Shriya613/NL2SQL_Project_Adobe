BEGIN TRANSACTION;
CREATE TABLE Hospital(
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Hospital" VALUES(1,'Western Medical Center');
INSERT INTO "Hospital" VALUES(2,'Eastern General Hospital');
INSERT INTO "Hospital" VALUES(3,'Northern Healthcare System');
INSERT INTO "Hospital" VALUES(4,'Central Valley Clinic');
CREATE TABLE JobPosting(
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    salary FLOAT,
    post_date DATE
);
INSERT INTO "JobPosting" VALUES(1,'Radiologist','Perform diagnostic imaging procedures.',80000.0,'2022-01-01');
INSERT INTO "JobPosting" VALUES(2,'Ultrasound Technician','Operate ultrasound equipment for medical imaging.',60000.0,'2022-01-02');
INSERT INTO "JobPosting" VALUES(3,'Radiographer','Produce X-ray images and other diagnostic images.',70000.0,'2022-01-03');
INSERT INTO "JobPosting" VALUES(4,'Diagnostic Medical Sonographer','Conduct ultrasound examinations on patients.',65000.0,'2022-01-04');
INSERT INTO "JobPosting" VALUES(5,'Healthcare Administrator','Manage healthcare facility operations and staff.',55000.0,'2022-01-05');
CREATE TABLE Radiologist(
    id INTEGER PRIMARY KEY,
    hospital_id INTEGER,
    FullName TEXT,
    FOREIGN KEY (hospital_id) REFERENCES Hospital(id)
);
INSERT INTO "Radiologist" VALUES(1,1,'John Doe');
INSERT INTO "Radiologist" VALUES(2,2,'Jane Smith');
INSERT INTO "Radiologist" VALUES(3,3,'Alice Brown');
INSERT INTO "Radiologist" VALUES(4,4,'Michael Johnson');
CREATE TABLE RadiologistJob(
    radiologist_id INTEGER,
    job_posting_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (radiologist_id) REFERENCES Radiologist(id),
    FOREIGN KEY (job_posting_id) REFERENCES JobPosting(id),
    PRIMARY KEY (radiologist_id, job_posting_id)
);
INSERT INTO "RadiologistJob" VALUES(1,1,'2022-01-15','2023-01-15');
INSERT INTO "RadiologistJob" VALUES(2,2,'2022-01-20','2023-01-20');
INSERT INTO "RadiologistJob" VALUES(3,3,'2022-01-25','2023-01-25');
INSERT INTO "RadiologistJob" VALUES(4,4,'2022-01-30','2023-01-30');
CREATE TABLE TechJob(
    tech_id INTEGER,
    job_posting_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (tech_id) REFERENCES Ultrasound_Tech(id),
    FOREIGN KEY (job_posting_id) REFERENCES JobPosting(id),
    PRIMARY KEY (tech_id, job_posting_id)
);
INSERT INTO "TechJob" VALUES(1,1,'2022-01-16','2023-01-16');
INSERT INTO "TechJob" VALUES(2,2,'2022-01-21','2023-01-21');
INSERT INTO "TechJob" VALUES(3,3,'2022-01-26','2023-01-26');
INSERT INTO "TechJob" VALUES(4,4,'2022-01-31','2023-01-31');
CREATE TABLE Ultrasound_Tech(
    id INTEGER PRIMARY KEY,
    hospital_id INTEGER,
    FullName TEXT,
    FOREIGN KEY (hospital_id) REFERENCES Hospital(id)
);
INSERT INTO "Ultrasound_Tech" VALUES(1,1,'Emma Jones');
INSERT INTO "Ultrasound_Tech" VALUES(2,2,'David Williams');
INSERT INTO "Ultrasound_Tech" VALUES(3,3,'Lisa Martin');
INSERT INTO "Ultrasound_Tech" VALUES(4,4,'Robert Thomas');
COMMIT;

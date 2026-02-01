BEGIN TRANSACTION;
CREATE TABLE Applicants (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone_number TEXT,
    job_id INTEGER,
    FOREIGN KEY(job_id) REFERENCES Jobs(id) ON DELETE CASCADE
);
INSERT INTO "Applicants" VALUES(1,'John','Doe','john.doe@example.com','555-123-4567',1);
INSERT INTO "Applicants" VALUES(2,'Jane','Smith','jane.smith@example.com','555-987-6543',1);
CREATE TABLE Clinics (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    zipcode TEXT
);
INSERT INTO "Clinics" VALUES(1,'Harmony Wellness Center','123 Main St','Anytown','CA','12345');
INSERT INTO "Clinics" VALUES(2,'Mental Health Solutions','456 Elm St','Another City','NY','67890');
CREATE TABLE Jobs (
    id INTEGER PRIMARY KEY,
    title TEXT,
    description TEXT,
    salary INTEGER,
    start_date DATE,
    end_date DATE,
    status TEXT 
);
INSERT INTO "Jobs" VALUES(1,'Psychologist','Providing therapy to children and adolescents.',60000,'2023-01-01','2023-12-31','Open');
INSERT INTO "Jobs" VALUES(2,'Psychiatrist','Diagnosing and treating mental disorders.',80000,'2023-01-01','2023-12-31','Open');
CREATE TABLE Psychiatrists (
    id INTEGER PRIMARY KEY,
    name TEXT,
    specialty TEXT,
    experience_years INTEGER,
    clinic TEXT,
    FOREIGN KEY(clinic) REFERENCES Clinics(name) ON DELETE SET NULL
);
INSERT INTO "Psychiatrists" VALUES(1,'Dr. Alice Brown','General Adult',8,'Harmony Wellness Center');
INSERT INTO "Psychiatrists" VALUES(2,'Dr. Bob Johnson','Geriatric',9,'Mental Health Associates');
CREATE TABLE Psychologists (
    id INTEGER PRIMARY KEY,
    name TEXT,
    specialty TEXT,
    experience_years INTEGER,
    clinic TEXT,
    FOREIGN KEY(clinic) REFERENCES Clinics(name) ON DELETE SET NULL
);
INSERT INTO "Psychologists" VALUES(1,'Dr. Jane Smith','Child & Adolescent',10,'Harmony Wellness Center');
INSERT INTO "Psychologists" VALUES(2,'Dr. John Doe','Cognitive Behavioral Therapy',7,'Mental Health Solutions');
COMMIT;

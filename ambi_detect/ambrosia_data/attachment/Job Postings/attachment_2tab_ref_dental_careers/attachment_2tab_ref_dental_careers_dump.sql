BEGIN TRANSACTION;
CREATE TABLE Education(
    id INTEGER PRIMARY KEY,
    degree_name TEXT);
INSERT INTO "Education" VALUES(1,'Doctor of Dental Surgery');
INSERT INTO "Education" VALUES(2,'Master of Science in Dentistry');
INSERT INTO "Education" VALUES(3,'Bachelor of Dental Surgery');
INSERT INTO "Education" VALUES(4,'Diploma in Dental Technology');
INSERT INTO "Education" VALUES(5,'Associate Degree in Applied Sciences - Dental Assisting');
CREATE TABLE Orthodontists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    education_id INTEGER,
    specialty_id INTEGER,
    FOREIGN KEY (education_id) REFERENCES Education(id),
    FOREIGN KEY (specialty_id) REFERENCES Specialization(id));
INSERT INTO "Orthodontists" VALUES(1,'John Doe',1,1);
INSERT INTO "Orthodontists" VALUES(2,'Jane Smith',2,2);
INSERT INTO "Orthodontists" VALUES(3,'Michael Johnson',3,1);
INSERT INTO "Orthodontists" VALUES(4,'Emily Davis',4,2);
INSERT INTO "Orthodontists" VALUES(5,'William Brown',5,3);
CREATE TABLE Periodontists(
    id INTEGER PRIMARY KEY,
    name TEXT,
    education_id INTEGER,
    specialty_id INTEGER,
    FOREIGN KEY (education_id) REFERENCES Education(id),
    FOREIGN KEY (specialty_id) REFERENCES Specialization(id));
INSERT INTO "Periodontists" VALUES(1,'Anna Martinez',1,4);
INSERT INTO "Periodontists" VALUES(2,'David Miller',2,3);
INSERT INTO "Periodontists" VALUES(3,'Emma Wilson',3,5);
INSERT INTO "Periodontists" VALUES(4,'Charlie Thompson',4,4);
INSERT INTO "Periodontists" VALUES(5,'Rachel Taylor',5,1);
CREATE TABLE Specialization(
    id INTEGER PRIMARY KEY,
    speciality_name TEXT);
INSERT INTO "Specialization" VALUES(1,'Orthodontics');
INSERT INTO "Specialization" VALUES(2,'Prosthodontics');
INSERT INTO "Specialization" VALUES(3,'Endodontics');
INSERT INTO "Specialization" VALUES(4,'Oral and Maxillofacial Surgery');
INSERT INTO "Specialization" VALUES(5,'Pediatric Dentistry');
COMMIT;

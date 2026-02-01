BEGIN TRANSACTION;
CREATE TABLE Applicants(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    phone TEXT);
INSERT INTO "Applicants" VALUES(1,'Alexander','Anderson','alex@email.com','+1234567890');
INSERT INTO "Applicants" VALUES(2,'Laura','Garcia','laura@email.com','+0987654321');
CREATE TABLE ApplicationsSubmitted(
    application_id INTEGER PRIMARY KEY AUTOINCREMENT,
    applicant_id INTEGER REFERENCES Applicants(id) ON DELETE CASCADE,
    job_posting_id INTEGER REFERENCES JobsPosted(id) ON DELETE CASCADE,
    resume TEXT,
    cover_letter TEXT);
INSERT INTO "ApplicationsSubmitted" VALUES(1,1,1,'resume1.pdf','cover_letter1.txt');
INSERT INTO "ApplicationsSubmitted" VALUES(2,2,2,'resume2.pdf','cover_letter2.txt');
CREATE TABLE Archivists(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INT,
    experience INT,
    location TEXT);
INSERT INTO "Archivists" VALUES(1,'Michael Johnson',28,7,'University''s Main Library');
INSERT INTO "Archivists" VALUES(2,'Emily Davis',33,6,'British Museum');
CREATE TABLE Employers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name TEXT,
    address TEXT,
    contact_info TEXT);
INSERT INTO "Employers" VALUES(1,'ABC Company','123 Street, City','+1234567890');
INSERT INTO "Employers" VALUES(2,'XYZ Company','456 Avenue, Town','+0987654321');
CREATE TABLE JobsPosted(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employer_id INTEGER REFERENCES Employers(id) ON DELETE SET NULL,
    position TEXT,
    description TEXT,
    start_date DATE,
    end_date DATE);
INSERT INTO "JobsPosted" VALUES(1,1,'Librarian','Responsible for managing library resources.','2022-01-01','2023-12-31');
INSERT INTO "JobsPosted" VALUES(2,2,'Curator','Manage museum exhibits and collections.','2022-07-01','2023-06-30');
CREATE TABLE Librarians(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INT,
    experience INT,
    location TEXT);
INSERT INTO "Librarians" VALUES(1,'John Doe',30,2,'University''s Main Library');
INSERT INTO "Librarians" VALUES(2,'Jane Smith',31,5,'New York Public Library');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Librarians',2);
INSERT INTO "sqlite_sequence" VALUES('Archivists',2);
INSERT INTO "sqlite_sequence" VALUES('Employers',2);
INSERT INTO "sqlite_sequence" VALUES('Applicants',2);
INSERT INTO "sqlite_sequence" VALUES('JobsPosted',2);
INSERT INTO "sqlite_sequence" VALUES('ApplicationsSubmitted',2);
COMMIT;

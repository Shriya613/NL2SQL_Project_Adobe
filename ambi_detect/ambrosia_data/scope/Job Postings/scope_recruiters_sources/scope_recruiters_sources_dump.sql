BEGIN TRANSACTION;
CREATE TABLE Recruiters (
    recruiter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Recruiters" VALUES(1,'John Doe','johndoe@example.com');
INSERT INTO "Recruiters" VALUES(2,'Jane Smith','janesmith@example.com');
INSERT INTO "Recruiters" VALUES(3,'Alice Johnson','alicejohnson@example.com');
INSERT INTO "Recruiters" VALUES(4,'Bob Brown','bobbrown@example.com');
INSERT INTO "Recruiters" VALUES(5,'Charlie Green','charliegreen@example.com');
CREATE TABLE Recruiters_Sources (
    recruiter_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recruiter_id INTEGER,
    source_id INTEGER,
    FOREIGN KEY(recruiter_id) REFERENCES Recruiters(recruiter_id),
    FOREIGN KEY(source_id) REFERENCES Sources(source_id)
);
INSERT INTO "Recruiters_Sources" VALUES(1,1,1);
INSERT INTO "Recruiters_Sources" VALUES(2,2,1);
INSERT INTO "Recruiters_Sources" VALUES(3,3,2);
INSERT INTO "Recruiters_Sources" VALUES(4,4,3);
INSERT INTO "Recruiters_Sources" VALUES(5,5,4);
INSERT INTO "Recruiters_Sources" VALUES(6,3,1);
INSERT INTO "Recruiters_Sources" VALUES(7,4,1);
INSERT INTO "Recruiters_Sources" VALUES(8,5,1);
CREATE TABLE Sources (
    source_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT
);
INSERT INTO "Sources" VALUES(1,'LinkedIn');
INSERT INTO "Sources" VALUES(2,'Indeed');
INSERT INTO "Sources" VALUES(3,'Glassdoor');
INSERT INTO "Sources" VALUES(4,'Monster');
INSERT INTO "Sources" VALUES(5,'CareerBuilder');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Recruiters',5);
INSERT INTO "sqlite_sequence" VALUES('Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Recruiters_Sources',8);
COMMIT;

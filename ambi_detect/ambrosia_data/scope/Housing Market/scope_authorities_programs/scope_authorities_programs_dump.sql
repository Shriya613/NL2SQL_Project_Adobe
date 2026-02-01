BEGIN TRANSACTION;
CREATE TABLE Authorities (
    authority_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    contact_info TEXT
);
INSERT INTO "Authorities" VALUES(1,'Authority A','contact@authoritya.com');
INSERT INTO "Authorities" VALUES(2,'Authority B','contact@authorityb.com');
INSERT INTO "Authorities" VALUES(3,'Authority C','contact@authorityc.com');
INSERT INTO "Authorities" VALUES(4,'Authority D','contact@authorityd.com');
INSERT INTO "Authorities" VALUES(5,'Authority E','contact@authoritye.com');
CREATE TABLE Authorities_Programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    authority_id INTEGER,
    program_id INTEGER,
    FOREIGN KEY(authority_id) REFERENCES Authorities(authority_id),
    FOREIGN KEY(program_id) REFERENCES Programs(program_id)
);
INSERT INTO "Authorities_Programs" VALUES(1,1,1);
INSERT INTO "Authorities_Programs" VALUES(2,2,1);
INSERT INTO "Authorities_Programs" VALUES(3,3,1);
INSERT INTO "Authorities_Programs" VALUES(4,4,1);
INSERT INTO "Authorities_Programs" VALUES(5,5,1);
INSERT INTO "Authorities_Programs" VALUES(6,5,5);
CREATE TABLE Programs (
    program_id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT,
    description TEXT
);
INSERT INTO "Programs" VALUES(1,'Public_Housing','A government-subsidized housing program.');
INSERT INTO "Programs" VALUES(2,'Affordable_Housing','Low-cost housing options for low-income families.');
INSERT INTO "Programs" VALUES(3,'Rental_Assistance','Financial assistance for rent payments.');
INSERT INTO "Programs" VALUES(4,'Homeownership_Incentives','Grants and loans for first-time home buyers.');
INSERT INTO "Programs" VALUES(5,'Emergency_Housing','Temporary housing solutions for those facing homelessness.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authorities',5);
INSERT INTO "sqlite_sequence" VALUES('Programs',5);
INSERT INTO "sqlite_sequence" VALUES('Authorities_Programs',6);
COMMIT;

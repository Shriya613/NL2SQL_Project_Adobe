BEGIN TRANSACTION;
CREATE TABLE Instruments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instrumentName TEXT,
    description TEXT);
INSERT INTO "Instruments" VALUES(1,'Microscope','A device used for viewing objects at high magnification.');
INSERT INTO "Instruments" VALUES(2,'Centrifuge','An apparatus that uses centrifugal force to separate substances based on their density.');
INSERT INTO "Instruments" VALUES(3,'Autoclave','A sterilization chamber that uses pressurized steam to kill bacteria and other microorganisms.');
INSERT INTO "Instruments" VALUES(4,'pH Meter','A device that measures the acidity or alkalinity of a solution.');
INSERT INTO "Instruments" VALUES(5,'Thermometer','A device that measures temperature.');
CREATE TABLE Lab_Technicians(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Lab_Technicians" VALUES(1,'John Doe','john.doe@example.com');
INSERT INTO "Lab_Technicians" VALUES(2,'Jane Smith','jane.smith@example.com');
INSERT INTO "Lab_Technicians" VALUES(3,'Alice Johnson','alice.johnson@example.com');
INSERT INTO "Lab_Technicians" VALUES(4,'Bob Brown','bob.brown@example.com');
INSERT INTO "Lab_Technicians" VALUES(5,'Charlie Davis','charlie.davis@example.com');
CREATE TABLE Lab_Technicians_Instruments(
    technicianId INTEGER,
    instrumentId INTEGER,
    FOREIGN KEY (technicianId) REFERENCES Lab_Technicians(id),
    FOREIGN KEY (instrumentId) REFERENCES Instruments(id));
INSERT INTO "Lab_Technicians_Instruments" VALUES(1,1);
INSERT INTO "Lab_Technicians_Instruments" VALUES(2,1);
INSERT INTO "Lab_Technicians_Instruments" VALUES(3,1);
INSERT INTO "Lab_Technicians_Instruments" VALUES(4,1);
INSERT INTO "Lab_Technicians_Instruments" VALUES(5,1);
INSERT INTO "Lab_Technicians_Instruments" VALUES(5,3);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Lab_Technicians',5);
INSERT INTO "sqlite_sequence" VALUES('Instruments',5);
COMMIT;

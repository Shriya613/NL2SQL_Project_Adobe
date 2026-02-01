BEGIN TRANSACTION;
CREATE TABLE Equipment(
    equipmentId INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT);
INSERT INTO "Equipment" VALUES(1,'Jet_Bridge');
INSERT INTO "Equipment" VALUES(2,'Luggage_Conveyor');
INSERT INTO "Equipment" VALUES(3,'Security_Scanner');
INSERT INTO "Equipment" VALUES(4,'Aircraft_Tow_Vehicle');
INSERT INTO "Equipment" VALUES(5,'Passenger_Board_bridge');
CREATE TABLE Maintenance_Facilities(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT);
INSERT INTO "Maintenance_Facilities" VALUES(1,'Facility1','Location1');
INSERT INTO "Maintenance_Facilities" VALUES(2,'Facility2','Location2');
INSERT INTO "Maintenance_Facilities" VALUES(3,'Facility3','Location3');
INSERT INTO "Maintenance_Facilities" VALUES(4,'Facility4','Location4');
INSERT INTO "Maintenance_Facilities" VALUES(5,'Facility5','Location5');
CREATE TABLE Maintenance_Facilities_Equipment(
    maintenanceId INTEGER,
    equipmentTypeId INTEGER,
    FOREIGN KEY (maintenanceId) REFERENCES Maintenance_Facilities(id),
    FOREIGN KEY (equipmentTypeId) REFERENCES Equipment(equipmentId));
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(1,1);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(1,2);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(2,3);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(2,4);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(3,1);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(3,5);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(4,2);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(4,6);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(5,3);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(5,7);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(2,1);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(4,1);
INSERT INTO "Maintenance_Facilities_Equipment" VALUES(5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Maintenance_Facilities',5);
INSERT INTO "sqlite_sequence" VALUES('Equipment',5);
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE Dormitories (
    dormitory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT ,
    capacity INTEGER
);
INSERT INTO "Dormitories" VALUES(1,'Mixed_Dormitory',20);
INSERT INTO "Dormitories" VALUES(2,'Female Only',15);
INSERT INTO "Dormitories" VALUES(3,'Mixed_Dormitory',30);
INSERT INTO "Dormitories" VALUES(4,'Male Only',25);
INSERT INTO "Dormitories" VALUES(5,'Mixed_Dormitory',20);
INSERT INTO "Dormitories" VALUES(6,'Female Only',10);
INSERT INTO "Dormitories" VALUES(7,'Mixed_Dormitory',35);
INSERT INTO "Dormitories" VALUES(8,'Male Only',30);
INSERT INTO "Dormitories" VALUES(9,'Mixed_Dormitory',20);
INSERT INTO "Dormitories" VALUES(10,'Female Only',15);
CREATE TABLE Hostels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT
);
INSERT INTO "Hostels" VALUES(1,'Hostel A','123 Main St');
INSERT INTO "Hostels" VALUES(2,'Hostel B','456 Elm St');
INSERT INTO "Hostels" VALUES(3,'Hostel C','789 Oak St');
INSERT INTO "Hostels" VALUES(4,'Hostel D','012 Pine St');
INSERT INTO "Hostels" VALUES(5,'Hostel E','345 Maple Ave');
CREATE TABLE Hostels_Dormitories (
    hostel_dormitory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    hostel_id INTEGER REFERENCES Hostels(id) ON DELETE CASCADE,
    dormitory_id INTEGER REFERENCES Dormitories(dormitory_id) ON DELETE CASCADE
);
INSERT INTO "Hostels_Dormitories" VALUES(1,1,1);
INSERT INTO "Hostels_Dormitories" VALUES(2,1,2);
INSERT INTO "Hostels_Dormitories" VALUES(3,2,3);
INSERT INTO "Hostels_Dormitories" VALUES(4,2,4);
INSERT INTO "Hostels_Dormitories" VALUES(5,3,5);
INSERT INTO "Hostels_Dormitories" VALUES(6,3,6);
INSERT INTO "Hostels_Dormitories" VALUES(7,4,7);
INSERT INTO "Hostels_Dormitories" VALUES(8,4,8);
INSERT INTO "Hostels_Dormitories" VALUES(9,5,9);
INSERT INTO "Hostels_Dormitories" VALUES(10,5,10);
INSERT INTO "Hostels_Dormitories" VALUES(11,2,1);
INSERT INTO "Hostels_Dormitories" VALUES(12,3,1);
INSERT INTO "Hostels_Dormitories" VALUES(13,4,1);
INSERT INTO "Hostels_Dormitories" VALUES(14,5,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Hostels',5);
INSERT INTO "sqlite_sequence" VALUES('Dormitories',10);
INSERT INTO "sqlite_sequence" VALUES('Hostels_Dormitories',14);
COMMIT;

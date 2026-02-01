BEGIN TRANSACTION;
CREATE TABLE Convention_Centers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    capacity INTEGER);
INSERT INTO "Convention_Centers" VALUES(1,'Center A',1000);
INSERT INTO "Convention_Centers" VALUES(2,'Center B',2000);
INSERT INTO "Convention_Centers" VALUES(3,'Center C',3000);
INSERT INTO "Convention_Centers" VALUES(4,'Center D',4000);
INSERT INTO "Convention_Centers" VALUES(5,'Center E',5000);
CREATE TABLE Convention_Centers_Event_Spaces(
    convention_center_id INTEGER REFERENCES Convention_Centers(id) ON DELETE CASCADE,
    event_space_id INTEGER REFERENCES Event_Spaces(id));
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(1,1);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(1,2);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(2,1);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(2,2);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(3,1);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(3,2);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(4,1);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(4,2);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(5,1);
INSERT INTO "Convention_Centers_Event_Spaces" VALUES(5,2);
CREATE TABLE Event_Spaces(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    space_name TEXT,
    type TEXT );
INSERT INTO "Event_Spaces" VALUES(1,'Duty-Free Shopping','shop');
INSERT INTO "Event_Spaces" VALUES(2,'Exhibition Hall','hall');
INSERT INTO "Event_Spaces" VALUES(3,'Banquet Room','room');
INSERT INTO "Event_Spaces" VALUES(4,'Theater','theatre');
INSERT INTO "Event_Spaces" VALUES(5,'Lounge Area','lounge');
CREATE TABLE Guests(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT,
    email TEXT);
INSERT INTO "Guests" VALUES(1,'John','Smith','johnsmith@example.com');
INSERT INTO "Guests" VALUES(2,'Jane','Doe','janedoe@example.com');
INSERT INTO "Guests" VALUES(3,'Michael','Black','mblack@example.com');
INSERT INTO "Guests" VALUES(4,'Emily','White','emwhite@example.com');
INSERT INTO "Guests" VALUES(5,'David','Gray','dgray@example.com');
CREATE TABLE Reservations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guest_id INTEGER REFERENCES Guests(id) ON DELETE CASCADE,
    event_space_id INTEGER REFERENCES Event_Spaces(id) ON DELETE SET NULL,
    checkin DATETIME,
    checkout DATETIME);
INSERT INTO "Reservations" VALUES(1,1,1,'2021-05-01 10:00:00','2021-05-01 18:00:00');
INSERT INTO "Reservations" VALUES(2,2,2,'2021-05-02 10:00:00','2021-05-02 18:00:00');
INSERT INTO "Reservations" VALUES(3,3,3,'2021-05-03 10:00:00','2021-05-03 18:00:00');
INSERT INTO "Reservations" VALUES(4,4,4,'2021-05-04 10:00:00','2021-05-04 18:00:00');
INSERT INTO "Reservations" VALUES(5,5,5,'2021-05-05 10:00:00','2021-05-05 18:00:00');
CREATE TABLE Rooms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_number INTEGER,
    floor_no INTEGER,
    capacity INTEGER);
INSERT INTO "Rooms" VALUES(1,101,1,50);
INSERT INTO "Rooms" VALUES(2,102,1,60);
INSERT INTO "Rooms" VALUES(3,103,1,70);
INSERT INTO "Rooms" VALUES(4,201,2,80);
INSERT INTO "Rooms" VALUES(5,202,2,90);
INSERT INTO "Rooms" VALUES(6,203,2,100);
INSERT INTO "Rooms" VALUES(7,301,3,110);
INSERT INTO "Rooms" VALUES(8,302,3,120);
INSERT INTO "Rooms" VALUES(9,303,3,130);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Convention_Centers',5);
INSERT INTO "sqlite_sequence" VALUES('Event_Spaces',5);
INSERT INTO "sqlite_sequence" VALUES('Rooms',9);
INSERT INTO "sqlite_sequence" VALUES('Guests',5);
INSERT INTO "sqlite_sequence" VALUES('Reservations',5);
COMMIT;

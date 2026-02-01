BEGIN TRANSACTION;
CREATE TABLE Concert(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    date DATETIME,
    artist_name TEXT,
    venue_id INTEGER,
    FOREIGN KEY (venue_id) REFERENCES Venue(id));
INSERT INTO "Concert" VALUES(1,'Rock Show','2022-10-10 19:00:00','Band A',1);
INSERT INTO "Concert" VALUES(2,'Pop Concert','2022-11-10 20:00:00','Artist B',2);
INSERT INTO "Concert" VALUES(3,'Jazz Night','2022-12-10 21:00:00','Musician C',3);
INSERT INTO "Concert" VALUES(4,'Classical Symphony','2022-11-20 18:00:00','Composer D',4);
INSERT INTO "Concert" VALUES(5,'Metal Festival','2022-12-20 22:00:00','Group E',5);
CREATE TABLE Theater_Play(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    start_date DATETIME,
    end_date DATETIME,
    theater_company TEXT,
    venue_id INTEGER,
    FOREIGN KEY (venue_id) REFERENCES Venue(id));
INSERT INTO "Theater_Play" VALUES(1,'Shakespeare Play','2022-10-10 19:00:00','2022-10-20 22:00:00','Company F',1);
INSERT INTO "Theater_Play" VALUES(2,'Modern Drama','2022-11-10 20:00:00','2022-11-20 22:00:00','Company G',2);
INSERT INTO "Theater_Play" VALUES(3,'Children''s Musical','2022-12-10 18:00:00','2022-12-20 22:00:00','Company H',3);
INSERT INTO "Theater_Play" VALUES(4,'Historical Tragedy','2022-11-20 19:00:00','2022-12-20 22:00:00','Company I',4);
INSERT INTO "Theater_Play" VALUES(5,'Romantic Comedy','2022-12-10 20:00:00','2022-12-20 22:00:00','Company J',5);
CREATE TABLE Venue(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    venue_capacity INT);
INSERT INTO "Venue" VALUES(1,'Arena','123 Street',500);
INSERT INTO "Venue" VALUES(2,'Theatre','456 Avenue',800);
INSERT INTO "Venue" VALUES(3,'Cinema Hall','789 Boulevard',400);
INSERT INTO "Venue" VALUES(4,'Performing Arts Center','000 Lane',600);
INSERT INTO "Venue" VALUES(5,'Music Auditorium','999 Road',700);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Venue',5);
INSERT INTO "sqlite_sequence" VALUES('Concert',5);
INSERT INTO "sqlite_sequence" VALUES('Theater_Play',5);
COMMIT;

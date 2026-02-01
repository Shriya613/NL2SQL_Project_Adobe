BEGIN TRANSACTION;
CREATE TABLE Artists (
    artist_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    genre TEXT
);
INSERT INTO "Artists" VALUES(1,'Michael Jackson','Pop');
INSERT INTO "Artists" VALUES(2,'Elvis Presley','Rock and Roll');
INSERT INTO "Artists" VALUES(3,'Madonna','Pop');
INSERT INTO "Artists" VALUES(4,'Prince','Pop');
INSERT INTO "Artists" VALUES(5,'Beyonce','R&B');
CREATE TABLE Events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist_id INTEGER,
    venue_id INTEGER,
    event_date DATE,
    FOREIGN KEY(artist_id) REFERENCES Artists(artist_id),
    FOREIGN KEY(venue_id) REFERENCES Venues(venue_id)
);
INSERT INTO "Events" VALUES(1,1,1,'2023-08-17');
INSERT INTO "Events" VALUES(2,2,2,'2023-09-16');
INSERT INTO "Events" VALUES(3,3,3,'2023-10-07');
INSERT INTO "Events" VALUES(4,4,4,'2023-10-21');
INSERT INTO "Events" VALUES(5,5,5,'2023-11-05');
CREATE TABLE "Sale_Date" (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    start_time TIME,
    end_time TIME,
    FOREIGN KEY(event_id) REFERENCES Events(event_id)
);
INSERT INTO "Sale_Date" VALUES(1,1,'12:00','23:59');
INSERT INTO "Sale_Date" VALUES(2,2,'12:00','23:59');
INSERT INTO "Sale_Date" VALUES(3,3,'12:00','23:59');
INSERT INTO "Sale_Date" VALUES(4,4,'12:00','23:59');
INSERT INTO "Sale_Date" VALUES(5,5,'12:00','23:59');
CREATE TABLE "Tickets" (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    seat_number TEXT,
    price DECIMAL(10,2) ,
    ON_SALE_DATE_ID INTEGER,
    FOREIGN KEY(ON_SALE_DATE_ID) REFERENCES "Sale_Date"(sale_id),
    FOREIGN KEY(event_id) REFERENCES Events(event_id)
);
INSERT INTO "Tickets" VALUES(1,1,'A1',20,1);
INSERT INTO "Tickets" VALUES(2,1,'A2',20,1);
INSERT INTO "Tickets" VALUES(3,2,'A1',20,2);
INSERT INTO "Tickets" VALUES(4,3,'A1',20,3);
INSERT INTO "Tickets" VALUES(5,4,'A1',20,4);
INSERT INTO "Tickets" VALUES(6,5,'A1',20,5);
CREATE TABLE Venues (
    venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT
);
INSERT INTO "Venues" VALUES(1,'Wembley Stadium','London');
INSERT INTO "Venues" VALUES(2,'Staples Center','Los Angeles');
INSERT INTO "Venues" VALUES(3,'Nissan Stadium','Nashville');
INSERT INTO "Venues" VALUES(4,'York Arena','New York');
INSERT INTO "Venues" VALUES(5,'Hollywood Bowl','Los Angeles');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Artists',5);
INSERT INTO "sqlite_sequence" VALUES('Venues',5);
INSERT INTO "sqlite_sequence" VALUES('Events',5);
INSERT INTO "sqlite_sequence" VALUES('Sale_Date',5);
INSERT INTO "sqlite_sequence" VALUES('Tickets',6);
COMMIT;

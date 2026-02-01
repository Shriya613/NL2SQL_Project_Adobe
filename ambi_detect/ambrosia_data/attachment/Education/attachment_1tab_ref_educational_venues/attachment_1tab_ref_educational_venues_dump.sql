BEGIN TRANSACTION;
CREATE TABLE Educational_Venues(
    VenueID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Type TEXT ,
    Address TEXT,
    City TEXT,
    State TEXT);
INSERT INTO "Educational_Venues" VALUES(1,'National History Museum','Museum','123 Main St.','Washington','DC');
INSERT INTO "Educational_Venues" VALUES(2,'Smithsonian Air & Space Museum','Museum','600 Independence Ave SW','Washington','DC');
INSERT INTO "Educational_Venues" VALUES(3,'Adler Planetarium','Planetarium','1300 S Lake Shore Drive','Chicago','IL');
INSERT INTO "Educational_Venues" VALUES(4,'Field Museum of Natural History','Museum','1400 S DuSable Lake Shore Dr.','Chicago','IL');
INSERT INTO "Educational_Venues" VALUES(5,'The Franklin Institute Science Museum','Museum','222 N 20th St.','Philadelphia','PA');
CREATE TABLE Tickets(
    TicketID INTEGER PRIMARY KEY AUTOINCREMENT,
    Price REAL);
INSERT INTO "Tickets" VALUES(1,15.0);
INSERT INTO "Tickets" VALUES(2,20.0);
INSERT INTO "Tickets" VALUES(3,25.0);
INSERT INTO "Tickets" VALUES(4,30.0);
INSERT INTO "Tickets" VALUES(5,35.0);
CREATE TABLE Venue_Tickets(
    VenueID INTEGER,
    TicketID INTEGER,
    FOREIGN KEY(VenueID) REFERENCES Educational_Venues(VenueID),
    FOREIGN KEY(TicketID) REFERENCES Tickets(TicketID),
    PRIMARY KEY(VenueID, TicketID));
INSERT INTO "Venue_Tickets" VALUES(1,1);
INSERT INTO "Venue_Tickets" VALUES(1,2);
INSERT INTO "Venue_Tickets" VALUES(2,1);
INSERT INTO "Venue_Tickets" VALUES(2,3);
INSERT INTO "Venue_Tickets" VALUES(3,2);
CREATE TABLE Visit_Records(
    RecordID INTEGER PRIMARY KEY AUTOINCREMENT,
    VenueID INTEGER,
    VisitorID INTEGER,
    Date DATE,
    Time TIME,
    FOREIGN KEY(VenueID) REFERENCES Educational_Venues(VenueID),
    FOREIGN KEY(VisitorID) REFERENCES Visitors(VisitorID));
INSERT INTO "Visit_Records" VALUES(1,1,1,'2022-01-01','10:00:00');
INSERT INTO "Visit_Records" VALUES(2,1,2,'2022-01-01','14:00:00');
INSERT INTO "Visit_Records" VALUES(3,2,1,'2022-01-02','10:00:00');
INSERT INTO "Visit_Records" VALUES(4,3,1,'2022-01-03','10:00:00');
INSERT INTO "Visit_Records" VALUES(5,3,2,'2022-01-03','14:00:00');
CREATE TABLE Visitors(
    VisitorID INTEGER PRIMARY KEY AUTOINCREMENT,
    FirstName TEXT,
    LastName TEXT,
    Age INTEGER,
    Email TEXT UNIQUE);
INSERT INTO "Visitors" VALUES(1,'John','Doe',30,'john.doe@example.com');
INSERT INTO "Visitors" VALUES(2,'Jane','Doe',28,'jane.doe@example.com');
INSERT INTO "Visitors" VALUES(3,'Alice','Brown',25,'alice.brown@example.com');
INSERT INTO "Visitors" VALUES(4,'Bob','White',35,'bob.white@example.com');
INSERT INTO "Visitors" VALUES(5,'Carol','Green',40,'carol.green@example.com');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Educational_Venues',5);
INSERT INTO "sqlite_sequence" VALUES('Tickets',5);
INSERT INTO "sqlite_sequence" VALUES('Visitors',5);
INSERT INTO "sqlite_sequence" VALUES('Visit_Records',5);
COMMIT;

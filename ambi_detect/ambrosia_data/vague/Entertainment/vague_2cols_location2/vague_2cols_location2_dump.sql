BEGIN TRANSACTION;
CREATE TABLE Albums (
    AlbumID INTEGER PRIMARY KEY,
    Title TEXT,
    ReleaseYear INTEGER,
    ArtistID INTEGER,
    FOREIGN KEY(ArtistID) REFERENCES Artists(ArtistID)
);
INSERT INTO "Albums" VALUES(1,'Abbey Road',1969,1);
INSERT INTO "Albums" VALUES(2,'Thriller',1982,2);
INSERT INTO "Albums" VALUES(3,'21',2011,3);
CREATE TABLE Artists (
    ArtistID INTEGER PRIMARY KEY,
    Name TEXT,
    Genre TEXT
);
INSERT INTO "Artists" VALUES(1,'The Beatles','Rock');
INSERT INTO "Artists" VALUES(2,'Michael Jackson','Pop');
INSERT INTO "Artists" VALUES(3,'Adele','Pop');
CREATE TABLE Concerts (
    ConcertID INTEGER PRIMARY KEY,
    Date DATETIME,
    City TEXT,
    Country TEXT,
    VenueName TEXT,
    HeadlineArtistID INTEGER,
    FOREIGN KEY(HeadlineArtistID) REFERENCES Artists(ArtistID)
);
INSERT INTO "Concerts" VALUES(1,'2021-07-15 20:00:00','London','UK','Wembley Stadium',2);
INSERT INTO "Concerts" VALUES(2,'2021-08-01 20:00:00','New York','USA','MetLife Stadium',1);
INSERT INTO "Concerts" VALUES(3,'2021-09-10 20:00:00','Paris','France','Stade de France',3);
CREATE TABLE Customers (
    CustomerID INTEGER PRIMARY KEY,
    FirstName TEXT,
    LastName TEXT,
    Email TEXT,
    PhoneNumber VARCHAR(15),
    "AddressLine" TEXT,
    "PostalCode" TEXT);
INSERT INTO "Customers" VALUES(1,'John','Doe','johndoe@example.com','+1234567890','123 Main St','60601');
INSERT INTO "Customers" VALUES(2,'Jane','Smith','janesmith@example.com','+9876543210','456 Elm St','90001');
INSERT INTO "Customers" VALUES(3,'Emily','Williams','emilywilliams@example.com','+1357924680','789 Oak St','02110');
INSERT INTO "Customers" VALUES(4,'Peter','Jackson','pjackson@example.com','+19876543210','123 Maple Ave','19101');
CREATE TABLE Tickets (
    TicketID INTEGER PRIMARY KEY,
    PurchaseDate DATETIME,
    SeatNumber TEXT,
    CustomerID INTEGER,
    ConcertID INTEGER,
    Price DECIMAL(10,2),
    FOREIGN KEY(CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY(ConcertID) REFERENCES Concerts(ConcertID)
);
INSERT INTO "Tickets" VALUES(1,'2021-07-15 12:00:00','Section A, Row 1, Seat 10',1,1,100);
INSERT INTO "Tickets" VALUES(2,'2021-07-15 12:00:00','Section B, Row 1, Seat 10',2,1,75);
INSERT INTO "Tickets" VALUES(3,'2021-08-01 12:00:00','Section C, Row 1, Seat 10',3,2,150);
INSERT INTO "Tickets" VALUES(4,'2021-08-01 12:00:00','Section D, Row 1, Seat 10',4,2,100);
COMMIT;

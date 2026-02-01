BEGIN TRANSACTION;
CREATE TABLE Locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    LocationName TEXT UNIQUE
);
INSERT INTO "Locations" VALUES(1,'Times Square, New York');
INSERT INTO "Locations" VALUES(2,'Bryant Park, New York');
INSERT INTO "Locations" VALUES(3,'Central Park, New York');
CREATE TABLE "Magicians" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INTEGER,
    Street_Location TEXT,
    FOREIGN KEY(Street_Location) REFERENCES Locations(id)
);
INSERT INTO "Magicians" VALUES(1,'Harry Houdini',130,'1');
INSERT INTO "Magicians" VALUES(2,'David Copperfield',65,'2');
INSERT INTO "Magicians" VALUES(3,'Criss Angel',47,'3');
INSERT INTO "Magicians" VALUES(4,'Dynamo',39,'1');
INSERT INTO "Magicians" VALUES(5,'Penn & Teller',75,'2');
CREATE TABLE "Mimes" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Age INTEGER,
    Street_Location TEXT,
    FOREIGN KEY(Street_Location) REFERENCES Locations(id)
);
INSERT INTO "Mimes" VALUES(1,'Marcel Marceau',88,'1');
INSERT INTO "Mimes" VALUES(2,'Whitey McConnell',95,'2');
INSERT INTO "Mimes" VALUES(3,'Charlie Chaplin',130,'3');
INSERT INTO "Mimes" VALUES(4,'Eric Jan De Groot',70,'1');
INSERT INTO "Mimes" VALUES(5,'Leonard Pitts Jr.',68,'2');
CREATE TABLE Reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ReviewerName TEXT,
    Rating INTEGER ,
    Comment TEXT,
    ShowID INTEGER,
    FOREIGN KEY(ShowID) REFERENCES Shows(id)
);
INSERT INTO "Reviews" VALUES(1,'John Doe',5,'Great show! Loved it!',1);
INSERT INTO "Reviews" VALUES(2,'Jane Smith',4,'Enjoyed the magic tricks!',1);
INSERT INTO "Reviews" VALUES(3,'Alice Johnson',3,'Good performance but could have been better.',2);
INSERT INTO "Reviews" VALUES(4,'Bob Brown',2,'Not my cup of tea.',2);
INSERT INTO "Reviews" VALUES(5,'Emma Wilson',4,'Amazing mime performance!',3);
CREATE TABLE Shows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ShowTitle TEXT,
    StartTime DATETIME,
    EndTime DATETIME,
    LocationID INTEGER,
    MagicianID INTEGER,
    MimeID INTEGER,
    FOREIGN KEY(LocationID) REFERENCES Locations(id),
    FOREIGN KEY(MagicianID) REFERENCES Magicians(id),
    FOREIGN KEY(MimeID) REFERENCES Mimes(id)
);
INSERT INTO "Shows" VALUES(1,'The Magic of Harry Houdini','2022-11-01 19:00:00','2022-11-01 21:00:00',1,1,NULL);
INSERT INTO "Shows" VALUES(2,'A Night of Illusion','2022-11-02 19:00:00','2022-11-02 21:00:00',2,2,NULL);
INSERT INTO "Shows" VALUES(3,'The Art of Silence','2022-11-03 19:00:00','2022-11-03 21:00:00',3,3,4);
INSERT INTO "Shows" VALUES(4,'The World of Marvel','2022-11-04 19:00:00','2022-11-04 21:00:00',4,NULL,5);
INSERT INTO "Shows" VALUES(5,'Laughter and Wonder','2022-11-05 19:00:00','2022-11-05 21:00:00',5,4,6);
CREATE TABLE Tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    PurchaserName TEXT,
    ShowID INTEGER,
    SeatNumber INT,
    Price DECIMAL(10,2) ,
    FOREIGN KEY(ShowID) REFERENCES Shows(id)
);
INSERT INTO "Tickets" VALUES(1,'John Doe',1,1,25);
INSERT INTO "Tickets" VALUES(2,'Jane Smith',1,2,25);
INSERT INTO "Tickets" VALUES(3,'Alice Johnson',2,1,30);
INSERT INTO "Tickets" VALUES(4,'Bob Brown',2,2,30);
INSERT INTO "Tickets" VALUES(5,'Emma Wilson',3,1,35);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Locations',3);
INSERT INTO "sqlite_sequence" VALUES('Shows',5);
INSERT INTO "sqlite_sequence" VALUES('Tickets',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
INSERT INTO "sqlite_sequence" VALUES('Magicians',5);
INSERT INTO "sqlite_sequence" VALUES('Mimes',5);
COMMIT;

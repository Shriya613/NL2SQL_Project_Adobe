BEGIN TRANSACTION;
CREATE TABLE Guitarists (
    id INTEGER PRIMARY KEY,
    musician_id INTEGER,
    guitar_model_id INTEGER,
    FOREIGN KEY(musician_id) REFERENCES Musicians(id),
    FOREIGN KEY(guitar_model_id) REFERENCES Guitars(id)
);
INSERT INTO "Guitarists" VALUES(1,3,4);
INSERT INTO "Guitarists" VALUES(2,4,5);
INSERT INTO "Guitarists" VALUES(3,5,3);
CREATE TABLE Guitars (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    price REAL,
    manufacturer_id INTEGER,
    FOREIGN KEY(manufacturer_id) REFERENCES Manufacturers(id)
);
INSERT INTO "Guitars" VALUES(1,'Stratocaster',699.99,1);
INSERT INTO "Guitars" VALUES(2,'Telecaster',799.99,1);
INSERT INTO "Guitars" VALUES(3,'Les Paul',1099.99,2);
INSERT INTO "Guitars" VALUES(4,'Acoustic Guitar',299.99,3);
INSERT INTO "Guitars" VALUES(5,'Electric Acoustic',499.99,3);
CREATE TABLE Manufacturers (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Manufacturers" VALUES(1,'Bonners Music');
INSERT INTO "Manufacturers" VALUES(2,'Fender');
INSERT INTO "Manufacturers" VALUES(3,'Yamaha');
INSERT INTO "Manufacturers" VALUES(4,'C. F. Martin & Co.');
INSERT INTO "Manufacturers" VALUES(5,'Steinway & Sons');
CREATE TABLE Musicians (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE
);
INSERT INTO "Musicians" VALUES(1,'John Doe');
INSERT INTO "Musicians" VALUES(2,'Jane Smith');
INSERT INTO "Musicians" VALUES(3,'Alice Johnson');
INSERT INTO "Musicians" VALUES(4,'Bob Williams');
INSERT INTO "Musicians" VALUES(5,'Charlie Brown');
CREATE TABLE Pianists (
    id INTEGER PRIMARY KEY,
    musician_id INTEGER,
    piano_model_id INTEGER,
    FOREIGN KEY(musician_id) REFERENCES Musicians(id),
    FOREIGN KEY(piano_model_id) REFERENCES Pianos(id)
);
INSERT INTO "Pianists" VALUES(1,1,1);
INSERT INTO "Pianists" VALUES(2,2,3);
CREATE TABLE Pianos (
    id INTEGER PRIMARY KEY,
    model_name TEXT,
    price REAL,
    manufacturer_id INTEGER,
    FOREIGN KEY(manufacturer_id) REFERENCES Manufacturers(id)
);
INSERT INTO "Pianos" VALUES(1,'Grand Piano',8999.99,1);
INSERT INTO "Pianos" VALUES(2,'Upright Piano',4999.99,1);
INSERT INTO "Pianos" VALUES(3,'Digital Piano',999.99,3);
INSERT INTO "Pianos" VALUES(4,'Stage Piano',1499.99,5);
INSERT INTO "Pianos" VALUES(5,'Hybrid Piano',1999.99,5);
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE Improvised_Theatres(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Description TEXT
, `venueID` TEXT REFERENCES `Venues`(`id`));
INSERT INTO "Improvised_Theatres" VALUES(1,'Comedy Club','A popular venue for stand-up comedy shows.','1');
INSERT INTO "Improvised_Theatres" VALUES(2,'Open Stage','An intimate theatre where actors can perform their own material.','2');
CREATE TABLE "Musicals"(
    id INTEGER PRIMARY KEY AUTOINCREMENT, `name` TEXT,
    Composer TEXT,
    Lyricist TEXT
, `venueID` TEXT REFERENCES `Venues`(`id`));
INSERT INTO "Musicals" VALUES(1,'West Side Story','Stephen Sondheim','George Furth','1');
INSERT INTO "Musicals" VALUES(2,'The Phantom of the Opera','Andrew Lloyd Webber','Tim Rice','2');
CREATE TABLE Venues(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT UNIQUE
);
INSERT INTO "Venues" VALUES(1,'Cambridge Theatre');
INSERT INTO "Venues" VALUES(2,'Her Majesty''s Theatre');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Improvised_Theatres',2);
INSERT INTO "sqlite_sequence" VALUES('Venues',2);
INSERT INTO "sqlite_sequence" VALUES('Musicals',2);
COMMIT;

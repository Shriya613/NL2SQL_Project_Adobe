BEGIN TRANSACTION;
CREATE TABLE Advertisements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT
);
INSERT INTO "Advertisements" VALUES(1,'Summer Sale','Get up to 70% off on summer clothes.');
INSERT INTO "Advertisements" VALUES(2,'New Arrivals','Check out our latest collection of winter jackets.');
INSERT INTO "Advertisements" VALUES(3,'Holiday Specials','Celebrate this holiday season with these amazing deals!');
INSERT INTO "Advertisements" VALUES(4,'Health Fair','Join us at the annual health fair to learn about staying fit.');
INSERT INTO "Advertisements" VALUES(5,'Art Exhibition','Don''t miss the upcoming art exhibition featuring local artists.');
CREATE TABLE Advertisements_Formats (
    advertisement_id INTEGER,
    format_id INTEGER,
    FOREIGN KEY(advertisement_id) REFERENCES Advertisements(id),
    FOREIGN KEY(format_id) REFERENCES Formats(id),
    PRIMARY KEY(advertisement_id, format_id)
);
INSERT INTO "Advertisements_Formats" VALUES(1,1);
INSERT INTO "Advertisements_Formats" VALUES(2,1);
INSERT INTO "Advertisements_Formats" VALUES(3,1);
INSERT INTO "Advertisements_Formats" VALUES(4,1);
INSERT INTO "Advertisements_Formats" VALUES(5,1);
INSERT INTO "Advertisements_Formats" VALUES(5,5);
CREATE TABLE Formats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    format_name TEXT
);
INSERT INTO "Formats" VALUES(1,'Billboard');
INSERT INTO "Formats" VALUES(2,'Digital Screen');
INSERT INTO "Formats" VALUES(3,'Poster');
INSERT INTO "Formats" VALUES(4,'Wall Banner');
INSERT INTO "Formats" VALUES(5,'Window Sticker');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Formats',5);
INSERT INTO "sqlite_sequence" VALUES('Advertisements',5);
COMMIT;

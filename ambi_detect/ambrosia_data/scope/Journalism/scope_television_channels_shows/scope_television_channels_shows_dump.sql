BEGIN TRANSACTION;
CREATE TABLE Shows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    show_title TEXT,
    genre TEXT
);
INSERT INTO "Shows" VALUES(1,'Friends','Sitcom');
INSERT INTO "Shows" VALUES(2,'The Office','Comedy');
INSERT INTO "Shows" VALUES(3,'Breaking Bad','Crime Drama');
INSERT INTO "Shows" VALUES(4,'Game Of Thrones','Fantasy');
INSERT INTO "Shows" VALUES(5,'Stranger Things','Science Fiction');
CREATE TABLE Television_Channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_name TEXT,
    country TEXT
);
INSERT INTO "Television_Channels" VALUES(1,'Channel One','USA');
INSERT INTO "Television_Channels" VALUES(2,'Channel Two','UK');
INSERT INTO "Television_Channels" VALUES(3,'Channel Three','Canada');
INSERT INTO "Television_Channels" VALUES(4,'Channel Four','Australia');
INSERT INTO "Television_Channels" VALUES(5,'Channel Five','Germany');
CREATE TABLE Television_Channels_Shows (
    television_channels_id INTEGER,
    shows_id INTEGER,
    FOREIGN KEY(television_channels_id) REFERENCES Television_Channels(id),
    FOREIGN KEY(shows_id) REFERENCES Shows(id)
);
INSERT INTO "Television_Channels_Shows" VALUES(1,1);
INSERT INTO "Television_Channels_Shows" VALUES(2,1);
INSERT INTO "Television_Channels_Shows" VALUES(3,1);
INSERT INTO "Television_Channels_Shows" VALUES(4,1);
INSERT INTO "Television_Channels_Shows" VALUES(5,1);
INSERT INTO "Television_Channels_Shows" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Television_Channels',5);
INSERT INTO "sqlite_sequence" VALUES('Shows',5);
COMMIT;

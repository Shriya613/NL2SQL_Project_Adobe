BEGIN TRANSACTION;
CREATE TABLE Content_Creators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Content_Creators" VALUES(1,'Alice','alice@example.com');
INSERT INTO "Content_Creators" VALUES(2,'Bob','bob@example.com');
INSERT INTO "Content_Creators" VALUES(3,'Charlie','charlie@example.com');
INSERT INTO "Content_Creators" VALUES(4,'Dave','dave@example.com');
INSERT INTO "Content_Creators" VALUES(5,'Eve','eve@example.com');
CREATE TABLE Content_Creators_Platforms (
    content_creator_id INTEGER,
    platform_id INTEGER,
    FOREIGN KEY (content_creator_id) REFERENCES Content_Creators(id),
    FOREIGN KEY (platform_id) REFERENCES Platforms(id),
    PRIMARY KEY (content_creator_id, platform_id)
);
INSERT INTO "Content_Creators_Platforms" VALUES(1,1);
INSERT INTO "Content_Creators_Platforms" VALUES(2,1);
INSERT INTO "Content_Creators_Platforms" VALUES(3,1);
INSERT INTO "Content_Creators_Platforms" VALUES(4,1);
INSERT INTO "Content_Creators_Platforms" VALUES(5,1);
INSERT INTO "Content_Creators_Platforms" VALUES(5,3);
CREATE TABLE Platforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_name TEXT
);
INSERT INTO "Platforms" VALUES(1,'YouTube');
INSERT INTO "Platforms" VALUES(2,'Twitch');
INSERT INTO "Platforms" VALUES(3,'Vimeo');
INSERT INTO "Platforms" VALUES(4,'Facebook');
INSERT INTO "Platforms" VALUES(5,'Instagram');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Content_Creators',5);
INSERT INTO "sqlite_sequence" VALUES('Platforms',5);
COMMIT;

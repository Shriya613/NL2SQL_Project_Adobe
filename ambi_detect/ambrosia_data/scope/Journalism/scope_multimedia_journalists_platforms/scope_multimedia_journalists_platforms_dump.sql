BEGIN TRANSACTION;
CREATE TABLE Multimedia_Journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT
);
INSERT INTO "Multimedia_Journalists" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Multimedia_Journalists" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Multimedia_Journalists" VALUES(3,'Michael Johnson','michael@example.com');
INSERT INTO "Multimedia_Journalists" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Multimedia_Journalists" VALUES(5,'William Brown','william@example.com');
CREATE TABLE Multimedia_Journalists_Platforms (
    multimedia_journalist_id INTEGER,
    platforms_id INTEGER,
    FOREIGN KEY(multimedia_journalist_id) REFERENCES Multimedia_Journalists(id),
    FOREIGN KEY(platforms_id) REFERENCES Platforms(id),
    PRIMARY KEY(multimedia_journalist_id, platforms_id)
);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(1,1);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(2,1);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(3,2);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(4,3);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(5,4);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(1,5);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(2,6);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(3,7);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(4,8);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(5,9);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(3,1);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(4,1);
INSERT INTO "Multimedia_Journalists_Platforms" VALUES(5,1);
CREATE TABLE Platforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    platform_name TEXT
);
INSERT INTO "Platforms" VALUES(1,'Video');
INSERT INTO "Platforms" VALUES(2,'Audio');
INSERT INTO "Platforms" VALUES(3,'Print');
INSERT INTO "Platforms" VALUES(4,'Online');
INSERT INTO "Platforms" VALUES(5,'Photo');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Multimedia_Journalists',5);
INSERT INTO "sqlite_sequence" VALUES('Platforms',5);
COMMIT;

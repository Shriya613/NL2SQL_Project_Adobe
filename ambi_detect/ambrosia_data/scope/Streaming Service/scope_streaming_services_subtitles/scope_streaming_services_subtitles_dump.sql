BEGIN TRANSACTION;
CREATE TABLE Streaming_Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','https://www.netflix.com');
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video','https://www.amazon.com/prime-video');
INSERT INTO "Streaming_Services" VALUES(3,'Hulu','https://www.hulu.com');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','https://www.disneyplus.com');
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max','https://www.hbomax.com');
CREATE TABLE Streaming_Services_Subtitles (
    streaming_service_id INTEGER,
    subtitle_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY(subtitle_id) REFERENCES Subtitles(id)
);
INSERT INTO "Streaming_Services_Subtitles" VALUES(1,1);
INSERT INTO "Streaming_Services_Subtitles" VALUES(2,2);
INSERT INTO "Streaming_Services_Subtitles" VALUES(3,3);
INSERT INTO "Streaming_Services_Subtitles" VALUES(4,4);
INSERT INTO "Streaming_Services_Subtitles" VALUES(5,5);
INSERT INTO "Streaming_Services_Subtitles" VALUES(2,1);
INSERT INTO "Streaming_Services_Subtitles" VALUES(3,1);
INSERT INTO "Streaming_Services_Subtitles" VALUES(4,1);
INSERT INTO "Streaming_Services_Subtitles" VALUES(5,1);
CREATE TABLE Subtitles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language TEXT,
    default_subtitle BOOLEAN DEFAULT FALSE
);
INSERT INTO "Subtitles" VALUES(1,'English',1);
INSERT INTO "Subtitles" VALUES(2,'Spanish',0);
INSERT INTO "Subtitles" VALUES(3,'French',0);
INSERT INTO "Subtitles" VALUES(4,'German',0);
INSERT INTO "Subtitles" VALUES(5,'Italian',0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Subtitles',5);
COMMIT;

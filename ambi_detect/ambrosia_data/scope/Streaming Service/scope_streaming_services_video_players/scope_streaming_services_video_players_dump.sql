BEGIN TRANSACTION;
CREATE TABLE Streaming_Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    description TEXT
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','A popular streaming platform offering movies, TV shows, documentaries.');
INSERT INTO "Streaming_Services" VALUES(2,'Hulu','An American subscription video on-demand service offering a library of films and television series.');
INSERT INTO "Streaming_Services" VALUES(3,'Amazon Prime Video','A subsidiary of Amazon that offers digital video streamings of movies and TV shows.');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','A subscription-based streaming service from Walt Disney direct-to-consumer.');
INSERT INTO "Streaming_Services" VALUES(5,'Apple TV+','Apple’s own streaming service featuring original content by Apple Studios.');
CREATE TABLE Streaming_Services_Video_Players (
    streaming_service_id INTEGER,
    video_player_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY (video_player_id) REFERENCES Video_Players(id)
);
INSERT INTO "Streaming_Services_Video_Players" VALUES(1,1);
INSERT INTO "Streaming_Services_Video_Players" VALUES(1,2);
INSERT INTO "Streaming_Services_Video_Players" VALUES(2,1);
INSERT INTO "Streaming_Services_Video_Players" VALUES(3,2);
INSERT INTO "Streaming_Services_Video_Players" VALUES(4,1);
INSERT INTO "Streaming_Services_Video_Players" VALUES(3,1);
INSERT INTO "Streaming_Services_Video_Players" VALUES(5,1);
CREATE TABLE Video_Players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_type TEXT,
    version INT
);
INSERT INTO "Video_Players" VALUES(1,'HTML5',3);
INSERT INTO "Video_Players" VALUES(2,'Flash',10);
INSERT INTO "Video_Players" VALUES(3,'Silverlight',5);
INSERT INTO "Video_Players" VALUES(4,'RealPlayer',2);
INSERT INTO "Video_Players" VALUES(5,'Windows Media Player',12);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Video_Players',5);
COMMIT;

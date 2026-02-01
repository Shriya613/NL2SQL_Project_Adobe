BEGIN TRANSACTION;
CREATE TABLE Streaming_Services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','A leading provider of on-demand internet streaming');
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video','Online streaming service from Amazon');
INSERT INTO "Streaming_Services" VALUES(3,'Hulu','Subscription video on demand service offering a selection of films and TV series');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','Streaming service featuring movies, shows, and original content from Disney');
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max','Premium streaming platform with HBO originals and library content');
CREATE TABLE Streaming_Services_Third_Party_Apps (
    streaming_service_id INTEGER,
    third_party_app_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY (third_party_app_id) REFERENCES Third_Party_Apps(app_id),
    PRIMARY KEY (streaming_service_id, third_party_app_id)
);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(1,1);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(2,3);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(3,2);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(4,4);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(5,5);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(2,1);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(3,1);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(4,1);
INSERT INTO "Streaming_Services_Third_Party_Apps" VALUES(5,1);
CREATE TABLE Third_Party_Apps (
    app_id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name TEXT,
    is_active BOOLEAN DEFAULT FALSE
);
INSERT INTO "Third_Party_Apps" VALUES(1,'Spotify_Connect',1);
INSERT INTO "Third_Party_Apps" VALUES(2,'Roku',1);
INSERT INTO "Third_Party_Apps" VALUES(3,'Google Chromecast',1);
INSERT INTO "Third_Party_Apps" VALUES(4,'Apple TV',1);
INSERT INTO "Third_Party_Apps" VALUES(5,'Fire TV',1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Third_Party_Apps',5);
COMMIT;

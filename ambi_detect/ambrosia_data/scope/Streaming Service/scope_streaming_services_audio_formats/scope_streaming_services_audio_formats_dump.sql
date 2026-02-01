BEGIN TRANSACTION;
CREATE TABLE Audio_Formats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    format_name TEXT,
    description TEXT
);
INSERT INTO "Audio_Formats" VALUES(1,'MP3','A common digital audio format.');
INSERT INTO "Audio_Formats" VALUES(2,'WAV','An uncompressed PCM audio file format.');
INSERT INTO "Audio_Formats" VALUES(3,'FLAC','Lossless compression audio file format.');
INSERT INTO "Audio_Formats" VALUES(4,'DTS','Digital Theater System sound codec.');
INSERT INTO "Audio_Formats" VALUES(5,'Dolby_Atmos','Object-based surround sound system.');
CREATE TABLE Streaming_Services(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    service_name TEXT,
    subscription_price REAL,
    number_of_users INTEGER
);
INSERT INTO "Streaming_Services" VALUES(1,'Service1',9.99,10000);
INSERT INTO "Streaming_Services" VALUES(2,'Service2',14.99,8000);
INSERT INTO "Streaming_Services" VALUES(3,'Service3',6.99,7000);
INSERT INTO "Streaming_Services" VALUES(4,'Service4',12.99,9000);
INSERT INTO "Streaming_Services" VALUES(5,'Service5',19.99,5000);
CREATE TABLE Streaming_Services_Audio_Formats(
    streaming_service_id INTEGER,
    audio_format_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Services(id),
    FOREIGN KEY (audio_format_id) REFERENCES Audio_Formats(id)
);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(1,5);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(2,5);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(3,5);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(4,5);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(5,5);
INSERT INTO "Streaming_Services_Audio_Formats" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Audio_Formats',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
COMMIT;

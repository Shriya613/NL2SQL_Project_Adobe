BEGIN TRANSACTION;
CREATE TABLE "Composer" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_date DATE
);
INSERT INTO "Composer" VALUES(1,'John Doe','1980-06-23');
INSERT INTO "Composer" VALUES(2,'Jane Smith','1978-05-10');
INSERT INTO "Composer" VALUES(3,'Michael Johnson','1990-07-21');
INSERT INTO "Composer" VALUES(4,'Emily Davis','1985-09-03');
INSERT INTO "Composer" VALUES(5,'William Brown','1992-11-04');
CREATE TABLE Fan (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE
);
INSERT INTO "Fan" VALUES(1,'Alice Cooper','alice@example.com');
INSERT INTO "Fan" VALUES(2,'Bob Smith','bob@example.com');
INSERT INTO "Fan" VALUES(3,'Carol Wilson','carol@example.com');
INSERT INTO "Fan" VALUES(4,'David Johnson','david@example.com');
INSERT INTO "Fan" VALUES(5,'Elizabeth Thompson','elizabeth@example.com');
CREATE TABLE Performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE,
    location TEXT,
    song_id INTEGER,
    FOREIGN KEY(song_id) REFERENCES Song(id)
);
INSERT INTO "Performance" VALUES(1,'2022-06-20','Central Park',1);
INSERT INTO "Performance" VALUES(2,'2022-07-10','Downtown Plaza',2);
INSERT INTO "Performance" VALUES(3,'2022-08-15','Concert Hall',3);
INSERT INTO "Performance" VALUES(4,'2022-09-05','Stadium',4);
INSERT INTO "Performance" VALUES(5,'2022-10-03','Amphitheater',5);
CREATE TABLE Song (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    duration NUMERIC,
    soundtrack_id INTEGER,
    FOREIGN KEY(soundtrack_id) REFERENCES Soundtrack(id)
);
INSERT INTO "Song" VALUES(1,'Dancing Through The Stars','3:30',1);
INSERT INTO "Song" VALUES(2,'Night Rider','4:00',2);
INSERT INTO "Song" VALUES(3,'Blue Groove','3:15',3);
INSERT INTO "Song" VALUES(4,'Time To Shine','3:45',4);
INSERT INTO "Song" VALUES(5,'Symphony In C','8:00',5);
CREATE TABLE "Soundtrack" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    musician_id INTEGER,
    FOREIGN KEY(musician_id) REFERENCES "Composer"(id)
);
INSERT INTO "Soundtrack" VALUES(1,'Music Festival','2022-06-25',1);
INSERT INTO "Soundtrack" VALUES(2,'Summer Jams','2022-07-09',2);
INSERT INTO "Soundtrack" VALUES(3,'Blues Night','2022-08-15',3);
INSERT INTO "Soundtrack" VALUES(4,'Party Time!','2022-09-01',4);
INSERT INTO "Soundtrack" VALUES(5,'Orchestral Showcase','2022-10-05',5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Composer',5);
INSERT INTO "sqlite_sequence" VALUES('Song',5);
INSERT INTO "sqlite_sequence" VALUES('Performance',5);
INSERT INTO "sqlite_sequence" VALUES('Fan',5);
INSERT INTO "sqlite_sequence" VALUES('Soundtrack',5);
COMMIT;

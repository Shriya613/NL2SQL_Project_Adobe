BEGIN TRANSACTION;
CREATE TABLE CirclePerformers(
    circle_id INTEGER REFERENCES Circles(id) ON DELETE CASCADE,
    performer_id INTEGER REFERENCES Performers(id) ON DELETE CASCADE,
    PRIMARY KEY (circle_id, performer_id));
INSERT INTO "CirclePerformers" VALUES(1,1);
INSERT INTO "CirclePerformers" VALUES(1,2);
INSERT INTO "CirclePerformers" VALUES(2,1);
INSERT INTO "CirclePerformers" VALUES(2,3);
INSERT INTO "CirclePerformers" VALUES(NULL,5);
INSERT INTO "CirclePerformers" VALUES(3,1);
CREATE TABLE CircleShows(
    circle_id INTEGER REFERENCES Circles(id) ON DELETE CASCADE,
    show_id INTEGER REFERENCES Shows(id) ON DELETE CASCADE,
    FOREIGN KEY (circle_id, show_id) REFERENCES Circles_Performers(circle_id, performer_id));
INSERT INTO "CircleShows" VALUES(1,1);
INSERT INTO "CircleShows" VALUES(1,2);
INSERT INTO "CircleShows" VALUES(2,1);
INSERT INTO "CircleShows" VALUES(2,2);
INSERT INTO "CircleShows" VALUES(3,3);
CREATE TABLE Circles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Circles" VALUES(1,'The Red Nose Circus');
INSERT INTO "Circles" VALUES(2,'The Magic Show');
INSERT INTO "Circles" VALUES(3,'Acrobatics Extravaganza');
CREATE TABLE Performers(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT );
INSERT INTO "Performers" VALUES(1,'Clown');
INSERT INTO "Performers" VALUES(2,'Magician');
INSERT INTO "Performers" VALUES(3,'Juggler');
INSERT INTO "Performers" VALUES(4,'Singer');
INSERT INTO "Performers" VALUES(5,'Dancer');
CREATE TABLE ShowVenues(
    show_id INTEGER REFERENCES Shows(id) ON DELETE CASCADE,
    venue_id INTEGER REFERENCES Venues(id) ON DELETE CASCADE,
    date DATE);
INSERT INTO "ShowVenues" VALUES(1,1,'2023-01-01');
INSERT INTO "ShowVenues" VALUES(2,2,'2023-02-16');
INSERT INTO "ShowVenues" VALUES(3,3,'2023-03-30');
CREATE TABLE Shows(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    start_date DATE,
    end_date DATE);
INSERT INTO "Shows" VALUES(1,'Opening Night','2023-01-01','2023-01-31');
INSERT INTO "Shows" VALUES(2,'Midweek Matinee','2023-02-15','2023-02-19');
INSERT INTO "Shows" VALUES(3,'End Of Season Party','2023-03-30','2023-03-31');
CREATE TABLE Venues(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Venues" VALUES(1,'City Hall Auditorium');
INSERT INTO "Venues" VALUES(2,'Big Top Tent');
INSERT INTO "Venues" VALUES(3,'Riverfront Amphitheater');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Circles',3);
INSERT INTO "sqlite_sequence" VALUES('Performers',5);
INSERT INTO "sqlite_sequence" VALUES('Shows',3);
INSERT INTO "sqlite_sequence" VALUES('Venues',3);
COMMIT;

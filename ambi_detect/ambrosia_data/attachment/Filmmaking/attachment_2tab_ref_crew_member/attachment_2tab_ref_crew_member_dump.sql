BEGIN TRANSACTION;
CREATE TABLE Actor(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT);
INSERT INTO "Actor" VALUES(1,'Harrison Ford');
INSERT INTO "Actor" VALUES(2,'Tom Cruise');
CREATE TABLE Award(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    award_type TEXT,
    film_id INTEGER,
    year INT,
    FOREIGN KEY(film_id) REFERENCES Film(id));
INSERT INTO "Award" VALUES(1,'Best Picture',1,1961);
INSERT INTO "Award" VALUES(2,'Best Direction',1,1961);
INSERT INTO "Award" VALUES(3,'Best Adapted Screenplay',1,1961);
INSERT INTO "Award" VALUES(4,'Best Cinematography',3,1983);
INSERT INTO "Award" VALUES(5,'Best Visual Effects',1,1961);
CREATE TABLE Character(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name TEXT,
    film_id INTEGER,
    actor_id INTEGER,
    FOREIGN KEY(film_id) REFERENCES Film(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id));
INSERT INTO "Character" VALUES(1,'Deckard',1,1);
INSERT INTO "Character" VALUES(2,'Jack Torrance',2,2);
INSERT INTO "Character" VALUES(3,'Gaff',3,1);
CREATE TABLE Cinematographer(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INT);
INSERT INTO "Cinematographer" VALUES(1,'Roger Deakins',20);
INSERT INTO "Cinematographer" VALUES(2,'Emmanuel Lubezki',18);
CREATE TABLE Director(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    years_of_experience INT);
INSERT INTO "Director" VALUES(1,'Alfred Hitchcock',20);
INSERT INTO "Director" VALUES(2,'Stanley Kubrick',30);
CREATE TABLE Film(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INT,
    director_id INTEGER,
    cinematographer_id INTEGER,
    FOREIGN KEY(director_id) REFERENCES Director(id),
    FOREIGN KEY(cinematographer_id) REFERENCES Cinematographer(id));
INSERT INTO "Film" VALUES(1,'Psycho',1960,1,NULL);
INSERT INTO "Film" VALUES(2,'The Shining',1980,2,NULL);
INSERT INTO "Film" VALUES(3,'Blade Runner',1982,2,1);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Director',2);
INSERT INTO "sqlite_sequence" VALUES('Cinematographer',2);
INSERT INTO "sqlite_sequence" VALUES('Film',3);
INSERT INTO "sqlite_sequence" VALUES('Actor',2);
INSERT INTO "sqlite_sequence" VALUES('Character',3);
INSERT INTO "sqlite_sequence" VALUES('Award',5);
COMMIT;

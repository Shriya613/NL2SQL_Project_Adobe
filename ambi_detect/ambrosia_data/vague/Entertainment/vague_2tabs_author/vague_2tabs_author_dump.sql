BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_year INTEGER
);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio',1974);
INSERT INTO "Actor" VALUES(2,'Joseph Gordon-Levitt',1981);
INSERT INTO "Actor" VALUES(3,'Tom Hardy',1977);
INSERT INTO "Actor" VALUES(4,'Cate Blanchett',1969);
INSERT INTO "Actor" VALUES(5,'Kristen Scott Thomas',1960);
CREATE TABLE Cast (
    movie_id INTEGER,
    actor_id INTEGER,
    role TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Cast" VALUES(1,1,'Norman Bates');
INSERT INTO "Cast" VALUES(1,2,'Marion Crane');
INSERT INTO "Cast" VALUES(2,1,'Fredric Quentlin');
INSERT INTO "Cast" VALUES(2,3,'Brad Pitt');
INSERT INTO "Cast" VALUES(3,4,'Robert Angier');
CREATE TABLE Filmmaker (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_year INTEGER
);
INSERT INTO "Filmmaker" VALUES(1,'Alfred Hitchcock',1899);
INSERT INTO "Filmmaker" VALUES(2,'Quentin Tarantino',1963);
INSERT INTO "Filmmaker" VALUES(3,'Christopher Nolan',1970);
INSERT INTO "Filmmaker" VALUES(4,'Martin Scorsese',1942);
INSERT INTO "Filmmaker" VALUES(5,'Stanley Kubrick',1928);
CREATE TABLE Genre (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre_name TEXT UNIQUE
);
INSERT INTO "Genre" VALUES(1,'Thriller');
INSERT INTO "Genre" VALUES(2,'Crime');
INSERT INTO "Genre" VALUES(3,'Drama');
INSERT INTO "Genre" VALUES(4,'Science Fiction');
INSERT INTO "Genre" VALUES(5,'Action');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    filmmaker_id INTEGER,
    screenwriter_id INTEGER,
    FOREIGN KEY(filmmaker_id) REFERENCES Filmmaker(id),
    FOREIGN KEY(screenwriter_id) REFERENCES Screenwriter(id)
);
INSERT INTO "Movie" VALUES(1,'Psycho','1960-06-16',1,1);
INSERT INTO "Movie" VALUES(2,'Inglourious Basterds','2009-08-20',2,2);
INSERT INTO "Movie" VALUES(3,'The Prestige','2006-10-19',3,3);
INSERT INTO "Movie" VALUES(4,'Taxi Driver','1976-02-08',4,5);
INSERT INTO "Movie" VALUES(5,'Blade Runner','1982-06-25',4,6);
CREATE TABLE Screenwriter (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    birth_year INTEGER
);
INSERT INTO "Screenwriter" VALUES(1,'Joseph Stefano',1922);
INSERT INTO "Screenwriter" VALUES(2,'Quentin Tarantino',1963);
INSERT INTO "Screenwriter" VALUES(3,'Jonathan Nolan',1976);
INSERT INTO "Screenwriter" VALUES(4,'Paul Schrader',1946);
INSERT INTO "Screenwriter" VALUES(5,'David Webb Peoples',1941);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Filmmaker',5);
INSERT INTO "sqlite_sequence" VALUES('Screenwriter',5);
INSERT INTO "sqlite_sequence" VALUES('Movie',5);
INSERT INTO "sqlite_sequence" VALUES('Genre',5);
INSERT INTO "sqlite_sequence" VALUES('Actor',5);
COMMIT;

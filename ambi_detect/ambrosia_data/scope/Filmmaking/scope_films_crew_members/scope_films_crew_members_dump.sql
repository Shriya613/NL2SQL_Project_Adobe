BEGIN TRANSACTION;
CREATE TABLE Actors (
    actor_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT,
    last_name TEXT
);
INSERT INTO "Actors" VALUES(1,'Tim','Robinson');
INSERT INTO "Actors" VALUES(2,'Tom','Hanks');
INSERT INTO "Actors" VALUES(3,'Brad','Pitt');
INSERT INTO "Actors" VALUES(4,'Paul','Newman');
INSERT INTO "Actors" VALUES(5,'Leonardo','DiCaprio');
CREATE TABLE Crew_Members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    name TEXT
);
INSERT INTO "Crew_Members" VALUES(1,'Director','John Doe');
INSERT INTO "Crew_Members" VALUES(2,'Producer','Jane Smith');
INSERT INTO "Crew_Members" VALUES(3,'Screenwriter','Mike Johnson');
INSERT INTO "Crew_Members" VALUES(4,'Cinematographer','Emily Brown');
INSERT INTO "Crew_Members" VALUES(5,'Editor','David Wilson');
CREATE TABLE Film_Genres (
    film_genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre TEXT
);
INSERT INTO "Film_Genres" VALUES(1,'Drama');
INSERT INTO "Film_Genres" VALUES(2,'Romance');
INSERT INTO "Film_Genres" VALUES(3,'Crime');
INSERT INTO "Film_Genres" VALUES(4,'Thriller');
INSERT INTO "Film_Genres" VALUES(5,'Science Fiction');
CREATE TABLE Films (
    film_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_year INTEGER
);
INSERT INTO "Films" VALUES(1,'The Shawshank Redemption',1994);
INSERT INTO "Films" VALUES(2,'Forrest Gump',1994);
INSERT INTO "Films" VALUES(3,'Pulp Fiction',1994);
INSERT INTO "Films" VALUES(4,'Fight Club',1999);
INSERT INTO "Films" VALUES(5,'Inception',2010);
CREATE TABLE Films_Actors (
    film_id INTEGER,
    actor_id INTEGER,
    character TEXT,
    FOREIGN KEY(film_id) REFERENCES Films(film_id),
    FOREIGN KEY(actor_id) REFERENCES Actors(actor_id),
    PRIMARY KEY(film_id, actor_id)
);
CREATE TABLE Films_Crew_Members (
    film_id INTEGER,
    member_id INTEGER,
    FOREIGN KEY(film_id) REFERENCES Films(film_id),
    FOREIGN KEY(member_id) REFERENCES Crew_Members(member_id),
    PRIMARY KEY(film_id, member_id)
);
INSERT INTO "Films_Crew_Members" VALUES(1,1);
INSERT INTO "Films_Crew_Members" VALUES(2,1);
INSERT INTO "Films_Crew_Members" VALUES(3,1);
INSERT INTO "Films_Crew_Members" VALUES(4,1);
INSERT INTO "Films_Crew_Members" VALUES(5,1);
INSERT INTO "Films_Crew_Members" VALUES(5,4);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Crew_Members',5);
INSERT INTO "sqlite_sequence" VALUES('Films',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Genres',5);
INSERT INTO "sqlite_sequence" VALUES('Actors',5);
COMMIT;

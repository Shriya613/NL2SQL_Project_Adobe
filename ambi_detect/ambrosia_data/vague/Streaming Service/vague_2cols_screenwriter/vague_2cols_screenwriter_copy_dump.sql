BEGIN TRANSACTION;
CREATE TABLE Actors (
    actor_id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    birth_date DATE,
    nationality TEXT
);
INSERT INTO "Actors" VALUES(1,'Al Pacino',NULL,19400425,'American');
INSERT INTO "Actors" VALUES(2,'Tom Hanks',NULL,19560709,'American');
INSERT INTO "Actors" VALUES(3,'Leonardo DiCaprio',NULL,19741111,'American');
INSERT INTO "Actors" VALUES(4,'Brad Pitt',NULL,19631218,'American');
INSERT INTO "Actors" VALUES(5,'Denzel Washington',NULL,19541227,'American');
CREATE TABLE "Movies" (
    movie_id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INTEGER,
    genre TEXT,
    Screenwriter TEXT,
    film TEXT
);
INSERT INTO "Movies" VALUES(1,'The Godfather',1972,'Crime, Drama','Mario Puzo','The Godfather');
INSERT INTO "Movies" VALUES(2,'Forrest Gump',1994,'Drama, Romance','Winston Groom','Forrest Gump');
INSERT INTO "Movies" VALUES(3,'Inception',2010,'Action, Adventure','Unknown','Inception');
INSERT INTO "Movies" VALUES(4,'Pulp Fiction',1994,'Crime, Drama','Quentin Tarantino & Roger Avary','Pulp Fiction');
INSERT INTO "Movies" VALUES(5,'Jurassic Park',1993,'Adventure, Sci-Fi','Michael Crichton','Jurassic Park');
CREATE TABLE Subscribers (
    subscriber_id INTEGER PRIMARY KEY,
    username TEXT,
    email TEXT,
    password TEXT
);
INSERT INTO "Subscribers" VALUES(1,'user001','user001@gmail.com','password001');
INSERT INTO "Subscribers" VALUES(2,'user002','user002@yahoo.com','password002');
INSERT INTO "Subscribers" VALUES(3,'user003','user003@hotmail.com','password003');
INSERT INTO "Subscribers" VALUES(4,'user004','user004@outlook.com','password004');
INSERT INTO "Subscribers" VALUES(5,'user005','user005@aol.com','password005');
CREATE TABLE TVShows (
    tvshow_id INTEGER PRIMARY KEY,
    show_title TEXT,
    start_date DATE,
    end_date DATE,
    genre TEXT,
    episodes INT
);
INSERT INTO "TVShows" VALUES(1,'Breaking Bad',20080117,20130929,'Crime, Drama',62);
INSERT INTO "TVShows" VALUES(2,'Game of Thrones',20110417,20190519,'Fantasy, Adventure',73);
INSERT INTO "TVShows" VALUES(3,'Stranger Things',20160715,'Present','Horror, Mystery',42);
INSERT INTO "TVShows" VALUES(4,'Friends',19940922,20040506,'Comedy, Romance',236);
INSERT INTO "TVShows" VALUES(5,'Vikings',20131003,'Present','Historical drama',89);
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    actor_name TEXT,
    birthdate DATE,
    nationality TEXT
);
INSERT INTO "Actor" VALUES(1,'Leonardo DiCaprio','1974-11-11','American');
INSERT INTO "Actor" VALUES(2,'Morgan Freeman','1937-06-01','American');
INSERT INTO "Actor" VALUES(3,'Kate Winslet','1975-10-05','English');
INSERT INTO "Actor" VALUES(4,'Christoph Waltz','1956-10-04','German');
INSERT INTO "Actor" VALUES(5,'Idris Elba','1972-09-06','British');
CREATE TABLE Culture (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT
);
INSERT INTO "Culture" VALUES(1,'Mexican','Mexican culture is a rich tapestry of indigenous and Spanish influences.');
INSERT INTO "Culture" VALUES(2,'Irish','Irish culture, with its deep roots in Celtic traditions, is celebrated for its literature, folk music, and dances.');
INSERT INTO "Culture" VALUES(3,'Spanish','Spanish culture is distinguished by its passionate music, dance and diverse regional identities.');
INSERT INTO "Culture" VALUES(4,'Argentine','Argentine culture reflects a unique fusion of European influences and indigenous traditions.');
INSERT INTO "Culture" VALUES(5,'Brazilian','Brazilian culture is vibrant and diverse, reflecting a blend of indigenous, Portuguese, and African influences.');
CREATE TABLE Dance (
    id INTEGER PRIMARY KEY,
    dance_name TEXT,
    culture_id INTEGER REFERENCES Culture(id),
    geographical_area_id INTEGER REFERENCES Geographical_Area(id)
);
INSERT INTO "Dance" VALUES(1,'Tango',4,4);
INSERT INTO "Dance" VALUES(2,'Flamenco',3,3);
INSERT INTO "Dance" VALUES(3,'Ballet Folklorico',1,5);
INSERT INTO "Dance" VALUES(4,'Irish Step Dance',2,3);
INSERT INTO "Dance" VALUES(5,'Capoeira',5,4);
CREATE TABLE Director (
    id INTEGER PRIMARY KEY,
    director_name TEXT,
    birthplace TEXT,
    known_for TEXT
);
INSERT INTO "Director" VALUES(1,'Frank Darabont','Montreal, Canada','Shawshank Redemption');
INSERT INTO "Director" VALUES(2,'Quentin Tarantino','Knoxville, USA','Pulp Fiction');
INSERT INTO "Director" VALUES(3,'Steven Spielberg','Cincinnati, USA','Schindler''s List');
INSERT INTO "Director" VALUES(4,'Robert Zemeckis','Chicago, USA','Forrest Gump');
INSERT INTO "Director" VALUES(5,'Gus Van Sant','Louisville, USA','Good Will Hunting');
CREATE TABLE Geographical_Area (
    id INTEGER PRIMARY KEY,
    region TEXT
);
INSERT INTO "Geographical_Area" VALUES(1,'Africa');
INSERT INTO "Geographical_Area" VALUES(2,'Asia');
INSERT INTO "Geographical_Area" VALUES(3,'Europe');
INSERT INTO "Geographical_Area" VALUES(4,'Latin America');
INSERT INTO "Geographical_Area" VALUES(5,'North America');
CREATE TABLE Movie (
    id INTEGER PRIMARY KEY,
    movie_title TEXT,
    release_year INTEGER ,
    director_id INTEGER REFERENCES Director(id)
);
INSERT INTO "Movie" VALUES(1,'The Shawshank Redemption',1994,1);
INSERT INTO "Movie" VALUES(2,'Pulp Fiction',1994,2);
INSERT INTO "Movie" VALUES(3,'Schindler''s List',1993,3);
INSERT INTO "Movie" VALUES(4,'Forrest Gump',1994,4);
INSERT INTO "Movie" VALUES(5,'Good Will Hunting',1997,5);
CREATE TABLE Music (
    id INTEGER PRIMARY KEY,
    music_title TEXT,
    genre TEXT,
    cultural_influence INTEGER REFERENCES Culture(id)
);
INSERT INTO "Music" VALUES(1,'River Flows In You','Classical',1);
INSERT INTO "Music" VALUES(2,'Despacito','Pop',8);
INSERT INTO "Music" VALUES(3,'Viva La Vida','Alternative Rock',9);
INSERT INTO "Music" VALUES(4,'La Vie En Rose','Jazz',10);
INSERT INTO "Music" VALUES(5,'Bohemian Rhapsody','Progressive Rock',11);
COMMIT;

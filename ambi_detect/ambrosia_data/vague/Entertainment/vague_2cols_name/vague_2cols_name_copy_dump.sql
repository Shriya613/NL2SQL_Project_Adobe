BEGIN TRANSACTION;
CREATE TABLE Actors (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    dateOfBirth DATE,
    movieId INTEGER,
    FOREIGN KEY(movieId) REFERENCES Movies(id)
);
INSERT INTO "Actors" VALUES(1,'Tom','Hanks','1956-07-09',1);
INSERT INTO "Actors" VALUES(2,'Robin','Williams','1958-07-21',2);
INSERT INTO "Actors" VALUES(3,'John','Travolta','1974-02-18',3);
INSERT INTO "Actors" VALUES(4,'Leonardo','DiCaprio','1974-11-11',4);
INSERT INTO "Actors" VALUES(5,'Ellen','Page','1987-02-21',5);
CREATE TABLE "Artists" (
    id INTEGER PRIMARY KEY,
    birthdate DATE,
    Name TEXT,
    story TEXT
);
INSERT INTO "Artists" VALUES(1,'1967-02-20','Kurt Donald Cobain','Lead singer and guitarist of Nirvana.');
INSERT INTO "Artists" VALUES(2,'1964-12-23','Edward Louis Severson III','Lead singer of Pearl Jam.');
INSERT INTO "Artists" VALUES(3,'1967-03-17','William Patrick Corgan Jr.','Frontman and songwriter of The Smashing Pumpkins.');
INSERT INTO "Artists" VALUES(4,'1942-06-20','Brian Douglas Wilson','Founding member and leader of The Beach Boys.');
INSERT INTO "Artists" VALUES(5,'1955-04-30','Stanley Clarke','One of the most influential bass players in jazz fusion.');
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT,
    birthdate DATE,
    realName TEXT,
    alias TEXT,
    story TEXT
);
INSERT INTO "Directors" VALUES(1,'Frank','Darabont','1959-01-28','Francis Lawrence Darabont',NULL,'Known for his work on The Green Mile and The Mist.');
INSERT INTO "Directors" VALUES(2,'Robert','Zemeckis','1951-05-14','Robert Lee Zemeckis','Bob Z.','Known for directing Back to the Future and Forrest Gump.');
INSERT INTO "Directors" VALUES(3,'Quentin','Tarantino','1963-03-27','Quentin Jerome Tarantino','QT','Known for his unique style of filmmaking and memorable dialogue.');
INSERT INTO "Directors" VALUES(4,'Christopher','Nolan','1970-07-30','Christopher Edward Nolan','Chris','Known for his thought-provoking films such as Inception and Interstellar.');
INSERT INTO "Directors" VALUES(5,'Jonathan','Nolan','1976-09-11','Jonathan Rowland Nolan','Jony','Brother of Christopher Nolan and co-writer of many of his movies.');
CREATE TABLE Movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    releaseDate DATE,
    genre TEXT,
    directorId INTEGER,
    FOREIGN KEY(directorId) REFERENCES Directors(id)
);
INSERT INTO "Movies" VALUES(1,'The Shawshank Redemption','1994-09-23','Drama',1);
INSERT INTO "Movies" VALUES(2,'Forrest Gump','1994-06-23','Drama/Romance',2);
INSERT INTO "Movies" VALUES(3,'Pulp Fiction','1994-10-17','Crime/Thriller',3);
INSERT INTO "Movies" VALUES(4,'Inception','2010-07-16','Action/Sci-Fi',4);
INSERT INTO "Movies" VALUES(5,'Interstellar','2014-11-07','Adventure/Sci-Fi',5);
CREATE TABLE MusicAlbums (
    id INTEGER PRIMARY KEY,
    albumTitle TEXT,
    releaseYear INTEGER ,
    artistId INTEGER,
    FOREIGN KEY(artistId) REFERENCES Artists(id)
);
INSERT INTO "MusicAlbums" VALUES(1,'Nevermind',1991,1);
INSERT INTO "MusicAlbums" VALUES(2,'Ten',1991,2);
INSERT INTO "MusicAlbums" VALUES(3,'Mellon Collie and the Infinite Sadness',1995,3);
INSERT INTO "MusicAlbums" VALUES(4,'Siamese Dream',1993,4);
INSERT INTO "MusicAlbums" VALUES(5,'The Colour and the Shape',1996,5);
CREATE TABLE Roles (
    id INTEGER PRIMARY KEY,
    actorID INTEGER,
    movieID INTEGER,
    roleName TEXT,
    characterDescription TEXT,
    FOREIGN KEY(actorID) REFERENCES Actors(id),
    FOREIGN KEY(movieID) REFERENCES Movies(id)
);
INSERT INTO "Roles" VALUES(1,1,1,'Andy Dufresne','A banker who is falsely accused of murder.');
INSERT INTO "Roles" VALUES(2,2,2,'Benjamin Buford "Bubba" Blue','A man with a passion for shrimp.');
INSERT INTO "Roles" VALUES(3,3,3,'Vincent Vega','A hitman with an appreciation for good music.');
INSERT INTO "Roles" VALUES(4,4,4,'Cobb','A thief who steals ideas from people''s dreams.');
INSERT INTO "Roles" VALUES(5,5,5,'Cooper','A pilot and scientist trying to save humanity.');
COMMIT;

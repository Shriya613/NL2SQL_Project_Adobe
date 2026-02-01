BEGIN TRANSACTION;
CREATE TABLE Directed_Movies (
    director_id INTEGER,
    movie_id INTEGER,
    FOREIGN KEY(director_id) REFERENCES Directors(id),
    FOREIGN KEY(movie_id) REFERENCES Movie_Projects(id)
);
INSERT INTO "Directed_Movies" VALUES(1,1);
INSERT INTO "Directed_Movies" VALUES(2,2);
INSERT INTO "Directed_Movies" VALUES(3,3);
INSERT INTO "Directed_Movies" VALUES(4,4);
INSERT INTO "Directed_Movies" VALUES(5,5);
CREATE TABLE Directors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    director_name TEXT,
    birth_year INT
);
INSERT INTO "Directors" VALUES(1,'Samuel L Jackson',1948);
INSERT INTO "Directors" VALUES(2,'Denzel Washington',1954);
INSERT INTO "Directors" VALUES(3,'Tom Hanks',1956);
INSERT INTO "Directors" VALUES(4,'Robert De Niro',1943);
INSERT INTO "Directors" VALUES(5,'Anthony Hopkins',1937);
CREATE TABLE Film_Crews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crew_name TEXT,
    role TEXT
);
INSERT INTO "Film_Crews" VALUES(1,'Camera Team','Director of Photography');
INSERT INTO "Film_Crews" VALUES(2,'Sound Team','Boom Operator');
INSERT INTO "Film_Crews" VALUES(3,'Production Designers','Art Director');
INSERT INTO "Film_Crews" VALUES(4,'Costume Department','Costumer');
INSERT INTO "Film_Crews" VALUES(5,'Special Effects','Pyrotechnics Specialist');
CREATE TABLE Locations_Managers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    contact_info TEXT
);
INSERT INTO "Locations_Managers" VALUES(1,'John Doe','+1234567890');
INSERT INTO "Locations_Managers" VALUES(2,'Jane Smith','+9876543210');
INSERT INTO "Locations_Managers" VALUES(3,'Alice Johnson','+1112223330');
INSERT INTO "Locations_Managers" VALUES(4,'Bob Brown','+4445556660');
INSERT INTO "Locations_Managers" VALUES(5,'Charlie Davis','+7778889990');
CREATE TABLE Locations_Managers_Responsibilities (
    locations_manager_id INTEGER,
    responsibility_id INTEGER,
    FOREIGN KEY(locations_manager_id) REFERENCES Locations_Managers(id),
    FOREIGN KEY(responsibility_id) REFERENCES Responsibilities(id)
);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(1,1);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(2,1);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(3,1);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(4,1);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(5,1);
INSERT INTO "Locations_Managers_Responsibilities" VALUES(5,5);
CREATE TABLE Movie_Projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    budget REAL
);
INSERT INTO "Movie_Projects" VALUES(1,'The Big Adventure','2022-12-01',10000000.0);
INSERT INTO "Movie_Projects" VALUES(2,'Alien Invasion','2023-06-15',15000000.0);
INSERT INTO "Movie_Projects" VALUES(3,'Love Story','2024-02-14',5000000.0);
INSERT INTO "Movie_Projects" VALUES(4,'Action Thriller','2025-08-20',20000000.0);
INSERT INTO "Movie_Projects" VALUES(5,'Comedy Drama','2026-04-01',7000000.0);
CREATE TABLE Movies_Cast (
    movie_id INTEGER,
    actor_id INTEGER,
    character_name TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie_Projects(id),
    FOREIGN KEY(actor_id) REFERENCES Actors(id)
);
INSERT INTO "Movies_Cast" VALUES(1,1,'Hero');
INSERT INTO "Movies_Cast" VALUES(1,2,'Villain');
INSERT INTO "Movies_Cast" VALUES(2,3,'Lead Scientist');
INSERT INTO "Movies_Cast" VALUES(2,4,'Intergalactic Warrior');
INSERT INTO "Movies_Cast" VALUES(3,5,'Lonely Heart');
INSERT INTO "Movies_Cast" VALUES(3,6,'Business Tycoon');
INSERT INTO "Movies_Cast" VALUES(4,7,'Detective');
INSERT INTO "Movies_Cast" VALUES(4,8,'Gang Leader');
INSERT INTO "Movies_Cast" VALUES(5,9,'Small Town Girl');
INSERT INTO "Movies_Cast" VALUES(5,10,'Wise Old Man');
CREATE TABLE Responsibilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT
);
INSERT INTO "Responsibilities" VALUES(1,'Film Permit Acquisition');
INSERT INTO "Responsibilities" VALUES(2,'Location Scouting');
INSERT INTO "Responsibilities" VALUES(3,'Health & Safety Compliance');
INSERT INTO "Responsibilities" VALUES(4,'Contract Negotiation');
INSERT INTO "Responsibilities" VALUES(5,'Equipment Coordination');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Locations_Managers',5);
INSERT INTO "sqlite_sequence" VALUES('Responsibilities',5);
INSERT INTO "sqlite_sequence" VALUES('Film_Crews',5);
INSERT INTO "sqlite_sequence" VALUES('Movie_Projects',5);
INSERT INTO "sqlite_sequence" VALUES('Directors',5);
COMMIT;

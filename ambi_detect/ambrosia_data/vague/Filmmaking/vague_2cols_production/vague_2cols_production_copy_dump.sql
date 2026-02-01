BEGIN TRANSACTION;
CREATE TABLE Actor (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    date_of_birth DATE,
    gender CHAR(1) 
);
INSERT INTO "Actor" VALUES(1,'Leonardo','DiCaprio','1974-11-11','M');
INSERT INTO "Actor" VALUES(2,'Tom','Hardy','1977-09-15','M');
CREATE TABLE Character (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    actor_id INTEGER,
    character_name TEXT,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(actor_id) REFERENCES Actor(id)
);
INSERT INTO "Character" VALUES(1,1,1,'Cobb');
INSERT INTO "Character" VALUES(2,1,2,'Arthur');
CREATE TABLE "Movie" (
    id INTEGER PRIMARY KEY,
    title TEXT,
    release_year INT,
    genre TEXT,
    Production INTEGER,
    FOREIGN KEY(Production) REFERENCES Studio(id),
    FOREIGN KEY(Production) REFERENCES Producer(id)
);
INSERT INTO "Movie" VALUES(1,'Inception',2010,'Action/Thriller',1);
INSERT INTO "Movie" VALUES(2,'The Dark Knight Rises',2012,'Action/Adventure',1);
CREATE TABLE Producer (
    id INTEGER PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    birthdate DATE
);
INSERT INTO "Producer" VALUES(1,'Jonathan','Nolan','jonathann@gmail.com','1976-07-24');
INSERT INTO "Producer" VALUES(2,'Christopher','Nolan','christophern@gmail.com','1970-08-30');
CREATE TABLE Review (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    reviewer_id INTEGER,
    rating INT,
    comments TEXT,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(movie_id) REFERENCES Movie(id),
    FOREIGN KEY(reviewer_id) REFERENCES Actor(id)
);
INSERT INTO "Review" VALUES(1,1,1,9,'Great movie!','2021-12-15 12:00:00');
INSERT INTO "Review" VALUES(2,1,2,8,'Amazing visuals and storytelling','2021-12-16 12:00:00');
CREATE TABLE Studio (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address_line1 TEXT,
    address_line2 TEXT,
    phone_number TEXT
);
INSERT INTO "Studio" VALUES(1,'Warner Bros.','108 W 9th St',NULL,'+1-310-786-7711');
INSERT INTO "Studio" VALUES(2,'Disney Studios','500 S Buena Vista St',NULL,'+1-818-560-1000');
COMMIT;

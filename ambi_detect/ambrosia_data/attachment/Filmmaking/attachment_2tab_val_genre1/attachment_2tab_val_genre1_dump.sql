BEGIN TRANSACTION;
CREATE TABLE Action_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "Action_Movies" VALUES(1,'Mission Impossible','Brian De Palma',1996,'Tom Cruise');
INSERT INTO "Action_Movies" VALUES(2,'The Bourne Identity','Doug Liman',2002,'Matt Damon');
INSERT INTO "Action_Movies" VALUES(3,'Mad Max: Fury Road','George Miller',2015,'Tom Hardy');
INSERT INTO "Action_Movies" VALUES(4,'John Wick','Chad Stahelski & David Leitch',2014,'Keanu Reeves');
INSERT INTO "Action_Movies" VALUES(5,'Inception','Christopher Nolan',2010,'Leonardo DiCaprio');
CREATE TABLE Adventure_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "Adventure_Movies" VALUES(1,'Indiana Jones and the Raiders of the Lost Ark','Steven Spielberg',1981,'Harrison Ford');
INSERT INTO "Adventure_Movies" VALUES(2,'Jurassic Park','Steven Spielberg',1993,'Samuel L. Jackson');
INSERT INTO "Adventure_Movies" VALUES(3,'Avatar','James Cameron',2009,'Sam Worthington');
INSERT INTO "Adventure_Movies" VALUES(4,'Titanic','James Cameron',1997,'Leonardo DiCaprio');
INSERT INTO "Adventure_Movies" VALUES(5,'Lord of the Rings: The Fellowship of the Ring','Peter Jackson',2001,'Elijah Wood');
CREATE TABLE Animated_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Studio TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "Animated_Movies" VALUES(1,'Frozen','Walt Disney Animation Studios',2013,'Kristen Bell');
INSERT INTO "Animated_Movies" VALUES(2,'Toy Story','Pixar Animation Studios',1995,'Tom Hanks');
INSERT INTO "Animated_Movies" VALUES(3,'Shrek','DreamWorks Animation',2001,'Mike Myers');
INSERT INTO "Animated_Movies" VALUES(4,'Zootopia','Walt Disney Animation Studios',2016,'Ginnifer Goodwin');
INSERT INTO "Animated_Movies" VALUES(5,'Wallace & Gromit: The Curse of the Were-Rabbit','Aardman Animations',2005,'Peter Sallis');
CREATE TABLE Comedy_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "Comedy_Movies" VALUES(1,'Monty Python and the Holy Grail','Terry Gilliam & Terry Jones',1975,'Graham Chapman');
INSERT INTO "Comedy_Movies" VALUES(2,'Airplane!','Jim Abrahams, David Zucker & Jerry Zucker',1980,'Leslie Nielsen');
INSERT INTO "Comedy_Movies" VALUES(3,'Bridesmaids','Paul Feig',2011,'Kristin Wiig');
INSERT INTO "Comedy_Movies" VALUES(4,'Step Brothers','Adam McKay',2008,'Will Ferrell');
INSERT INTO "Comedy_Movies" VALUES(5,'Superbad','Greg Mottola',2007,'Michael Cera');
CREATE TABLE Horror_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "Horror_Movies" VALUES(1,'Psycho','Alfred Hitchcock',1960,'Anthony Perkins');
INSERT INTO "Horror_Movies" VALUES(2,'The Exorcist','William Friedkin',1973,'Ellen Burstyn');
INSERT INTO "Horror_Movies" VALUES(3,'The Shining','Stanley Kubrick',1980,'Jack Nicholson');
INSERT INTO "Horror_Movies" VALUES(4,'Get Out','Jordan Peele',2017,'Daniel Kaluuya');
INSERT INTO "Horror_Movies" VALUES(5,'It Follows','David Robert Mitchell',2014,'Maika Monroe');
CREATE TABLE SciFi_Movies(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Director TEXT,
    ReleaseYear INTEGER,
    Stars TEXT);
INSERT INTO "SciFi_Movies" VALUES(1,'Star Wars Episode IV - A New Hope','George Lucas',1977,'Mark Hamill');
INSERT INTO "SciFi_Movies" VALUES(2,'Back to the Future','Robert Zemeckis',1985,'Michael J. Fox');
INSERT INTO "SciFi_Movies" VALUES(3,'Interstellar','Christopher Nolan',2014,'Matthew McConaughey');
INSERT INTO "SciFi_Movies" VALUES(4,'Blade Runner','Ridley Scott',1982,'Harrison Ford');
INSERT INTO "SciFi_Movies" VALUES(5,'Alien','Ridley Scott',1979,'Sigourney Weaver');
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE Actors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Roles INT);
INSERT INTO "Actors" VALUES(1,'Leonardo DiCaprio',45,'USA',15);
INSERT INTO "Actors" VALUES(2,'Keanu Reeves',55,'Canada',20);
INSERT INTO "Actors" VALUES(3,'Denzel Washington',65,'USA',10);
INSERT INTO "Actors" VALUES(4,'Will Smith',50,'USA',7);
INSERT INTO "Actors" VALUES(5,'Tom Cruise',55,'USA',15);
INSERT INTO "Actors" VALUES(6,'Mark Wahlberg',48,'USA',25);
INSERT INTO "Actors" VALUES(7,'Scarlett Johansson',35,'USA',18);
INSERT INTO "Actors" VALUES(8,'Natalie Portman',40,'Israel',12);
INSERT INTO "Actors" VALUES(9,'Idris Elba',48,'UK',15);
INSERT INTO "Actors" VALUES(10,'Viola Davis',55,'USA',20);
CREATE TABLE Cinematographers(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Films_Cinematographed INT);
INSERT INTO "Cinematographers" VALUES(1,'Roger Deakins',75,'USA',15);
INSERT INTO "Cinematographers" VALUES(2,'Janusz Kaminski',65,'Poland',20);
INSERT INTO "Cinematographers" VALUES(3,'Wally Pfister',60,'USA',10);
INSERT INTO "Cinematographers" VALUES(4,'Eric Alan Edwards',55,'USA',7);
INSERT INTO "Cinematographers" VALUES(5,'Phedon Papamichael',50,'Greece',15);
INSERT INTO "Cinematographers" VALUES(6,'Mihai Malaimare Jr.',45,'Romania',25);
INSERT INTO "Cinematographers" VALUES(7,'Roberto Schaefer',55,'Switzerland',18);
INSERT INTO "Cinematographers" VALUES(8,'Rodrigo Prieto',50,'Mexico',12);
INSERT INTO "Cinematographers" VALUES(9,'Simon Duggan',48,'Australia',15);
INSERT INTO "Cinematographers" VALUES(10,'Stuart Dryburgh',60,'New Zealand',20);
CREATE TABLE Directors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Movies_Directed INT);
INSERT INTO "Directors" VALUES(1,'Tom Hanks',65,'USA',15);
INSERT INTO "Directors" VALUES(2,'Steven Spielberg',75,'USA',20);
INSERT INTO "Directors" VALUES(3,'Christopher Nolan',50,'UK',10);
INSERT INTO "Directors" VALUES(4,'Quentin Tarantino',60,'USA',7);
INSERT INTO "Directors" VALUES(5,'Martin Scorsese',80,'USA',15);
INSERT INTO "Directors" VALUES(6,'Clint Eastwood',90,'USA',25);
INSERT INTO "Directors" VALUES(7,'Ang Lee',65,'Taiwan',18);
INSERT INTO "Directors" VALUES(8,'Ava DuVernay',50,'USA',12);
INSERT INTO "Directors" VALUES(9,'Guillermo del Toro',55,'Mexico',15);
INSERT INTO "Directors" VALUES(10,'Kathryn Bigelow',70,'USA',20);
CREATE TABLE Editors(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Projects_Completed INT);
INSERT INTO "Editors" VALUES(1,'Thelma Schoonmaker',75,'USA',15);
INSERT INTO "Editors" VALUES(2,'Lee Smith',65,'England',20);
INSERT INTO "Editors" VALUES(3,'Paul Hirsch',68,'USA',10);
INSERT INTO "Editors" VALUES(4,'Anne V. Coates',90,'England',7);
INSERT INTO "Editors" VALUES(5,'James Cameron',65,'Canada',15);
INSERT INTO "Editors" VALUES(6,'Conrad Buff IV',55,'USA',25);
INSERT INTO "Editors" VALUES(7,'Stephen Mirrione',50,'USA',18);
INSERT INTO "Editors" VALUES(8,'Tim Squyres',48,'USA',12);
INSERT INTO "Editors" VALUES(9,'Jeffrey Ford AICE',55,'USA',15);
INSERT INTO "Editors" VALUES(10,'Tom Cross',40,'USA',20);
CREATE TABLE Producers(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Productions INT);
INSERT INTO "Producers" VALUES(1,'Harvey Weinstein',65,'USA',15);
INSERT INTO "Producers" VALUES(2,'Jack Nicholson',75,'USA',20);
INSERT INTO "Producers" VALUES(3,'Barbra Streisand',75,'USA',10);
INSERT INTO "Producers" VALUES(4,'George Clooney',55,'USA',7);
INSERT INTO "Producers" VALUES(5,'Steve Martin',75,'USA',15);
INSERT INTO "Producers" VALUES(6,'Oprah Winfrey',65,'USA',25);
INSERT INTO "Producers" VALUES(7,'Robert De Niro',75,'USA',18);
INSERT INTO "Producers" VALUES(8,'Julia Roberts',50,'USA',12);
INSERT INTO "Producers" VALUES(9,'Morgan Freeman',65,'USA',15);
INSERT INTO "Producers" VALUES(10,'Drew Barrymore',45,'USA',20);
CREATE TABLE Screenwriters(
    ID INTEGER PRIMARY KEY,
    Name TEXT,
    Age INT,
    Country TEXT,
    Number_Of_Projects_Completed INT);
INSERT INTO "Screenwriters" VALUES(1,'John Doe',30,'USA',15);
INSERT INTO "Screenwriters" VALUES(2,'Jane Smith',28,'UK',20);
INSERT INTO "Screenwriters" VALUES(3,'Alice Johnson',35,'Canada',10);
INSERT INTO "Screenwriters" VALUES(4,'Bob Brown',40,'Australia',7);
INSERT INTO "Screenwriters" VALUES(5,'Charlie Green',32,'Germany',15);
INSERT INTO "Screenwriters" VALUES(6,'David White',29,'France',25);
INSERT INTO "Screenwriters" VALUES(7,'Emma Black',31,'Italy',18);
INSERT INTO "Screenwriters" VALUES(8,'Fred Grey',37,'Spain',12);
INSERT INTO "Screenwriters" VALUES(9,'Grace Blue',27,'Japan',15);
INSERT INTO "Screenwriters" VALUES(10,'Hannah Red',33,'India',20);
COMMIT;

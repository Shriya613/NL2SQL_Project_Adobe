BEGIN TRANSACTION;
CREATE TABLE Concerts(
    Date DATE,
    Location TEXT,
    Headliner TEXT,
    Openers TEXT,
    Ticket_Price REAL
);
INSERT INTO "Concerts" VALUES('2022-01-01','New York City','Led Zeppelin','Queen',120.0);
INSERT INTO "Concerts" VALUES('2022-02-01','Los Angeles','Black Sabbath','Deep Purple',100.0);
INSERT INTO "Concerts" VALUES('2022-03-01','Chicago','AC/DC','Lynyrd Skynyrd',90.0);
INSERT INTO "Concerts" VALUES('2022-04-01','Houston','Metallica','Slayer',150.0);
INSERT INTO "Concerts" VALUES('2022-05-01','San Francisco','Tool','Alice in Chains',130.0);
CREATE TABLE Fans(
    Name TEXT,
    Age INTEGER,
    Preferred_Genre TEXT
);
INSERT INTO "Fans" VALUES('John Doe',25,'Rock');
INSERT INTO "Fans" VALUES('Jane Smith',30,'Jazz');
INSERT INTO "Fans" VALUES('Tom Brown',40,'Classic Rock');
INSERT INTO "Fans" VALUES('Emily Johnson',50,'Prog Rock');
INSERT INTO "Fans" VALUES('William Wilson',60,'Blues Rock');
CREATE TABLE Jazz_Bands(
    Band TEXT,
    Founding_Year INT,
    Genre TEXT,
    Lead_Singer TEXT,
    Instruments TEXT
);
INSERT INTO "Jazz_Bands" VALUES('Miles Davis Quintet',1965,'Jazz','Miles Davis','Trumpet');
INSERT INTO "Jazz_Bands" VALUES('John Coltrane Quartet',1972,'Jazz','John Coltrane','Saxophone');
INSERT INTO "Jazz_Bands" VALUES('Thelonious Monk Quartet',1953,'Jazz','Thelonious Monk','Piano');
INSERT INTO "Jazz_Bands" VALUES('Dave Brubeck Quartet',1950,'Jazz','Dave Brubeck','Piano');
INSERT INTO "Jazz_Bands" VALUES('Weather Report',1970,'Fusion','Joe Zawinul','Keyboards');
CREATE TABLE Music_Labels(
    Label_Name TEXT,
    Artists TEXT,
    Year_Established INTEGER
);
INSERT INTO "Music_Labels" VALUES('Columbia Records','Bob Dylan, Bruce Springsteen, Adele',1887);
INSERT INTO "Music_Labels" VALUES('Warner Bros. Records','Green Day, Tom Petty, Nirvana',1958);
INSERT INTO "Music_Labels" VALUES('Atlantic Records','Led Zeppelin, AC/DC, Lynyrd Skynyrd',1947);
INSERT INTO "Music_Labels" VALUES('EMI','Beatles, Pink Floyd, Coldplay',1923);
INSERT INTO "Music_Labels" VALUES('Universal Music Group','U2, Eminem, Lady Gaga',1934);
CREATE TABLE Rock_Bands(
    Band TEXT,
    Founding_Year INT,
    Genre TEXT,
    Lead_Guitarist TEXT,
    Drummer TEXT
);
INSERT INTO "Rock_Bands" VALUES('Led Zeppelin',1968,'Hard Rock','Jimmy Page','John Bonham');
INSERT INTO "Rock_Bands" VALUES('Queen',1970,'Progressive Rock','Brian May','Roger Taylor');
INSERT INTO "Rock_Bands" VALUES('Black Sabbath',1968,'Heavy Metal','Tony Iommi','Bill Ward');
INSERT INTO "Rock_Bands" VALUES('Deep Purple',1968,'Hard Rock','Ritchie Blackmore','Ian Paice');
INSERT INTO "Rock_Bands" VALUES('AC/DC',1973,'Hard Rock','Angus Young','Phil Rudd');
CREATE TABLE Venues(
    Name TEXT,
    Address TEXT,
    Capacity INTEGER
);
INSERT INTO "Venues" VALUES('Madison Square Garden','New York, NY',20000);
INSERT INTO "Venues" VALUES('Staples Center','Los Angeles, CA',19000);
INSERT INTO "Venues" VALUES('United Center','Chicago, IL',23500);
INSERT INTO "Venues" VALUES('NRG Stadium','Houston, TX',71600);
INSERT INTO "Venues" VALUES('Oracle Arena','Oakland, CA',19596);
COMMIT;

BEGIN TRANSACTION;
CREATE TABLE artists (
    id INTEGER PRIMARY KEY,
    name TEXT,
    genre TEXT
);
INSERT INTO "artists" VALUES(1,'The Beatles','Rock');
INSERT INTO "artists" VALUES(2,'Michael Jackson','Pop');
INSERT INTO "artists" VALUES(3,'Lady Gaga','Pop');
INSERT INTO "artists" VALUES(4,'Elton John','Classic Rock');
INSERT INTO "artists" VALUES(5,'Adele','Pop');
CREATE TABLE bookings (
    id INTEGER PRIMARY KEY,
    userId INTEGER,
    ticketId INTEGER,
    bookingDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT ,
    FOREIGN KEY(userId) REFERENCES users(id),
    FOREIGN KEY(ticketId) REFERENCES tickets(id)
);
INSERT INTO "bookings" VALUES(1,1,1,'2024-02-19 13:13:50','Confirmed');
INSERT INTO "bookings" VALUES(2,2,2,'2024-02-19 13:13:50','Confirmed');
INSERT INTO "bookings" VALUES(3,3,3,'2024-02-19 13:13:50','Confirmed');
INSERT INTO "bookings" VALUES(4,4,4,'2024-02-19 13:13:50','Confirmed');
INSERT INTO "bookings" VALUES(5,5,5,'2024-02-19 13:13:50','Confirmed');
CREATE TABLE concerts (
    id INTEGER PRIMARY KEY,
    artistId INTEGER,
    date DATE,
    city TEXT,
    concertHall TEXT,
    FOREIGN KEY(artistId) REFERENCES artists(id)
);
INSERT INTO "concerts" VALUES(1,1,'2023-06-01','New York','Madison Square Garden');
INSERT INTO "concerts" VALUES(2,2,'2023-07-15','Los Angeles','Staples Center');
INSERT INTO "concerts" VALUES(3,3,'2023-08-30','Chicago','United Center');
INSERT INTO "concerts" VALUES(4,4,'2023-09-10','Dallas','American Airlines Center');
INSERT INTO "concerts" VALUES(5,5,'2023-10-01','Boston','TD Garden');
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY,
    concertId INTEGER,
    price DECIMAL(5,2),
    quantity INTEGER,
    type TEXT, -- e.g., VIP, General Admission
    FOREIGN KEY(concertId) REFERENCES concerts(id)
);
INSERT INTO "tickets" VALUES(1,1,100,1000,'General Admission');
INSERT INTO "tickets" VALUES(2,1,200,500,'VIP');
INSERT INTO "tickets" VALUES(3,2,120,1000,'General Admission');
INSERT INTO "tickets" VALUES(4,2,220,500,'VIP');
INSERT INTO "tickets" VALUES(5,3,150,1000,'General Admission');
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT,
    email TEXT,
    passwordHash TEXT
);
INSERT INTO "users" VALUES(1,'user1','user1@example.com','passwordhash1');
INSERT INTO "users" VALUES(2,'user2','user2@example.com','passwordhash2');
INSERT INTO "users" VALUES(3,'user3','user3@example.com','passwordhash3');
INSERT INTO "users" VALUES(4,'user4','user4@example.com','passwordhash4');
INSERT INTO "users" VALUES(5,'user5','user5@example.com','passwordhash5');
CREATE TABLE venues (
    id INTEGER PRIMARY KEY,
    venueName TEXT,
    location TEXT,
    capacity INTEGER
);
INSERT INTO "venues" VALUES(1,'Madison Square Garden','New York',20000);
INSERT INTO "venues" VALUES(2,'Staples Center','Los Angeles',19000);
INSERT INTO "venues" VALUES(3,'United Center','Chicago',23000);
INSERT INTO "venues" VALUES(4,'American Airlines Center','Dallas',19000);
INSERT INTO "venues" VALUES(5,'TD Garden','Boston',19000);
COMMIT;

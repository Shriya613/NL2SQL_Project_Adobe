BEGIN TRANSACTION;
CREATE TABLE hotels(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    accommodation_type TEXT );
INSERT INTO "hotels" VALUES(1,'Grand Hotel','Downtown','Hotel');
INSERT INTO "hotels" VALUES(2,'Park Plaza','Uptown','Hotel');
INSERT INTO "hotels" VALUES(3,'Young Travelers Hostel','City Center','Youth Hostel');
INSERT INTO "hotels" VALUES(4,'Backpacker Inn','Suburbia','Youth Hostel');
INSERT INTO "hotels" VALUES(5,'Luxury Suites','Beachside','Hotel');
INSERT INTO "hotels" VALUES(6,'Mountain Retreat','Outskirts','Hotel');
CREATE TABLE reviews(
    user_name TEXT,
    review_text TEXT,
    rating INTEGER ,
    room_number INTEGER,
    FOREIGN KEY (room_number) REFERENCES rooms(room_number));
INSERT INTO "reviews" VALUES('John Doe','Great experience!',5,101);
INSERT INTO "reviews" VALUES('Jane Smith','Not what I expected.',3,201);
INSERT INTO "reviews" VALUES('Emma Brown','Fantastic stay!',5,301);
INSERT INTO "reviews" VALUES('Mike Johnson','Too expensive.',2,401);
INSERT INTO "reviews" VALUES('Laura Wilson','Awesome hostel!',4,501);
INSERT INTO "reviews" VALUES('William Clark','I will come back again!',5,601);
INSERT INTO "reviews" VALUES('Kate King','Not my cup of tea.',3,701);
CREATE TABLE rooms(
    hotel_id INTEGER,
    room_number INTEGER,
    price REAL,
    FOREIGN KEY (hotel_id) REFERENCES hotels(id));
INSERT INTO "rooms" VALUES(1,101,50.0);
INSERT INTO "rooms" VALUES(3,301,50.0);
INSERT INTO "rooms" VALUES(4,401,80.0);
INSERT INTO "rooms" VALUES(6,601,50.0);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('hotels',6);
COMMIT;

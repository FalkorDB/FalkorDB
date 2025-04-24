go:
	cc -I../json_h json_test.c -o json_test
	./json_test < examples/1.json  > json_test.output
	./json_test < examples/2.json >> json_test.output
	./json_test < examples/3.json >> json_test.output
	./json_test < examples/4.json >> json_test.output
	./json_test < examples/5.json >> json_test.output
	./json_test < examples/6.json >> json_test.output
	./json_test < examples/A.json >> json_test.output
	diff json_test.output examples/json_test_output.txt

distclean: purge

purge:
	rm json_test json_test.output

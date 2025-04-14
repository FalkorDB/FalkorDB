import os
from common import *
from index_utils import *


class testACL():
    def __init__(self):
        ACL_GRAPH_ADMIN_COMMANDS = """INFO CLIENT DBSIZE PING HELLO AUTH RESTORE
        DUMP DEL EXISTS UNLINK TYPE FLUSHALL TOUCH EXPIRE PEXPIREAT TTL PTTL
        EXPIRETIME RENAME RENAMENX SCAN DISCARD EXEC MULTI UNWATCH WATCH ECHO
        SLOWLOG WAIT WAITAOF GRAPH.INFO GRAPH.LIST GRAPH.QUERY GRAPH.RO_QUERY
        GRAPH.EXPLAIN GRAPH.PROFILE GRAPH.DELETE GRAPH.CONSTRAINT GRAPH.SLOWLOG
        GRAPH.BULK GRAPH.CONFIG CLUSTER COMMAND GRAPH.PASSWORD GRAPH.ACL SAVE
        """

        os.environ['ACL_GRAPH_ADMIN'] = ACL_GRAPH_ADMIN_COMMANDS.replace('\n', ' ')
        
        ACL_GRAPH_USER_COMMANDS = """INFO CLIENT DBSIZE PING HELLO AUTH RESTORE 
        DUMP DEL EXISTS UNLINK TYPE FLUSHALL TOUCH EXPIRE PEXPIREAT TTL PTTL 
        EXPIRETIME RENAME RENAMENX SCAN DISCARD EXEC MULTI UNWATCH WATCH ECHO 
        SLOWLOG WAIT WAITAOF GRAPH.INFO GRAPH.LIST GRAPH.QUERY GRAPH.RO_QUERY 
        GRAPH.EXPLAIN GRAPH.PROFILE GRAPH.DELETE GRAPH.CONSTRAINT GRAPH.SLOWLOG 
        GRAPH.BULK CLUSTER COMMAND GRAPH.PASSWORD"""
                
        os.environ['ACL_GRAPH_USER'] = ACL_GRAPH_USER_COMMANDS.replace('\n', ' ')

        ACL_GRAPH_READONLY_USER_COMMANDS = """INFO CLIENT DBSIZE PING HELLO AUTH 
        RESTORE DUMP DEL EXISTS UNLINK TYPE FLUSHALL TOUCH EXPIRE PEXPIREAT TTL PTTL 
        EXPIRETIME RENAME RENAMENX SCAN DISCARD EXEC MULTI UNWATCH WATCH ECHO SLOWLOG 
        WAIT WAITAOF GRAPH.INFO GRAPH.LIST GRAPH.RO_QUERY GRAPH.EXPLAIN GRAPH.PROFILE 
        GRAPH.CONSTRAINT GRAPH.SLOWLOG GRAPH.BULK GRAPH.CONFIG CLUSTER COMMAND"""

        os.environ['ACL_GRAPH_READONLY_USER'] = ACL_GRAPH_READONLY_USER_COMMANDS.replace('\n', ' ')

        self.env, self.db = Env()
        self.redis_con = self.env.getConnection()

        # set password 'pass' for user 'default'
        # permit access to the command 'GRAPH.ACL'
        self.db.execute_command("ACL", "SETUSER", "default", "on", ">pass",
                                    "+GRAPH.ACL")
       
    def get_user_commands(self, user_details):
        """
        Extracts and returns the list of commands from Redis ACL GETUSER response.
    
        Args:
            user_details (list): Raw response from Redis ACL GETUSER command
        
        Returns:
            list: Commands with their permissions (e.g., ['-@all', '+info', ...])
        """

        # Convert the flat list to a dictionary
        user_dict = dict(zip(user_details[::2], user_details[1::2]))
    
        # Get the commands string and split into individual permissions
        commands_str = user_dict.get('commands', '')
        return commands_str.split()

    def test01_use_graph_acl_to_create_users(self):
        """
            make sure we're able to use the GRAPH.ACL command to create new users
        """

        v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-admin", 
                                    "on", ">pass", "+@graph-admin")
        self.env.assertTrue(v == "OK")
        user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-admin")
        user_commands = self.get_user_commands(user_details)
        
        self.env.assertEqual(48, len(user_commands))
        self.env.assertTrue('+graph.password' in user_commands)
        self.env.assertTrue('+graph.acl' in user_commands)
        
        #-----------------------------------------------------------------------

        v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-user", 
                                    "on", ">pass", "+@graph-user")
        self.env.assertTrue(v == "OK")
        user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-user")
        user_commands = self.get_user_commands(user_details)
        self.env.assertEqual(45, len(user_commands))
        self.env.assertTrue('+graph.password' in user_commands)
        self.env.assertFalse('+graph.acl' in user_commands)
        
        #-----------------------------------------------------------------------

        v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-read-only",
                                     "on", ">pass", "+@graph-readonly-user")
        self.env.assertTrue(v == "OK")
        user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-read-only")
        user_commands = self.get_user_commands(user_details)
        self.env.assertEqual(43, len(user_commands))
        self.env.assertFalse('+graph.password' in user_commands)
        self.env.assertFalse('+graph.acl' in user_commands)
        self.env.assertFalse('+graph.query' in user_commands)
        self.env.assertTrue('+graph.ro_query' in user_commands)
        
        
    def test02_graph_acl_cant_change_global_admin(self):
        """
        make sure that GRAPH.ACL can't change the 'global' admin
        """

        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use GRAPH.ACL to change the global admin
            v = self.db.execute_command("GRAPH.ACL", "SETUSER", "default", "-@all")
            self.env.assertTrue(False, "should not be able to change the global admin")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("FAILED" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")

    def test03_graph_acl_cant_get_globbal_admin(self):
        """
        make sure 'global' admin isn't visible
        """

        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use GRAPH.ACL to read the global admin
            v = self.db.execute_command("GRAPH.ACL", "GETUSER", "default")
            self.env.assertTrue(False, "should not be able to get the global admin")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("FAILED" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")
            
    def test04_graph_acl_ignore_unautherized_perimssions(self):
        """
        make sure we can't grant un-autherized permissions
        """

        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use add ACL to myself using GRAPH.ACL
            v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-admin", "+ACL")
            self.env.assertTrue(v == "OK")
        finally:
            self.db.execute_command("AUTH", "default", "pass")
            user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-admin")
            user_commands = self.get_user_commands(user_details)
            self.env.assertFalse('acl' in user_commands)
            self.env.assertEqual(48, len(user_commands))

    def test05_graph_acl_ignotr_unautherized_permissinos_with_pipe(self):
        """
        make sure we can't grant un-autherized permissions with pipe
        e.g. +COMMAND|LIST
        """
        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use GRAPH.ACL to change the global admin
            v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-user", "+COMMAND|LIST")
            self.env.assertTrue(v == "OK")
        finally:
            self.db.execute_command("AUTH", "default", "pass")
            user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-user")
            user_commands = self.get_user_commands(user_details)
            self.env.assertFalse('command|list' in user_commands)
            self.env.assertEqual(46, len(user_commands))
            
    def test06_set_user_off(self):
        """
        make sure we can set user off
        """
        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use GRAPH.ACL to change the global admin
            v = self.db.execute_command("GRAPH.ACL", "SETUSER", "falkordb-read-only", "off")
            self.env.assertTrue(v == "OK")
        finally:
            self.db.execute_command("AUTH", "default", "pass")
            user_details = self.db.execute_command("GRAPH.ACL", "GETUSER", "falkordb-read-only")
            user_dict = dict(zip(user_details[::2], user_details[1::2]))
            flags= user_dict.get('flags', '')
            self.env.assertTrue('off' in flags)
  
    def test07_graph_acl_wrong_call(self):
        """
        make sure we get the right error when calling the command with wrong args
        """
        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use add ACL to myself using GRAPH.ACL
            v = self.db.execute_command("GRAPH.ACL", "SETUSER")
            self.env.assertTrue(False, "should throw arity error")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("wrong number of arguments" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")
            
        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use add ACL to myself using GRAPH.ACL
            v = self.db.execute_command("GRAPH.ACL", "GETUSER")
            self.env.assertTrue(False, "should throw arity error")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("wrong number of arguments" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")

        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use add ACL to myself using GRAPH.ACL
            v = self.db.execute_command("GRAPH.ACL", "SAVE" , "EXTRA_ARG")
            self.env.assertTrue(False, "should throw arity error")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("wrong number of arguments" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")

        try:
            v = self.db.execute_command("AUTH", "falkordb-admin", "pass") 
            self.env.assertTrue(v)
            # try to use add ACL to myself using GRAPH.ACL
            v = self.db.execute_command("GRAPH.ACL", "FOO")
            self.env.assertTrue(False, "should throw Unknown command")
        except redis.exceptions.ResponseError as e:
            self.env.assertTrue("Unknown command" in str(e))
        finally:
            self.db.execute_command("AUTH", "default", "pass")
          
    def test100_wrong_password_call(self):
        """
        make sure we get the right error when calling the command with wrong args
        """
        try:
           v = self.db.execute_command("GRAPH.PASSWORD", "ADD")  
           self.env.assertTrue(False, "should throw arity error")
        except redis.exceptions.ResponseError as e:
            self.env.assertContains("wrong number of arguments", str(e))    
        ## unknown command
        try:
           v = self.db.execute_command("GRAPH.PASSWORD", "FOO", "BAR")  
           self.env.assertTrue(False, "should throw unknown command error")
        except redis.exceptions.ResponseError as e:
           self.env.assertContains("Unknown sub-command", str(e))    
        
    def test101_add_password(self):
        """
        make sure we can add a password to a user
        """
        v = self.db.execute_command("AUTH", "falkordb-user", "pass") 
        self.env.assertTrue(v)
        # try to use GRAPH.ACL to change the global admin
        v = self.db.execute_command("GRAPH.PASSWORD", "ADD", "foo")
        self.env.assertTrue(v == "OK")
        self.db.execute_command("AUTH", "default", "pass")
        self.env.assertTrue(v)
        self.db.execute_command("AUTH", "falkordb-user", "foo")
        self.env.assertTrue(v)
        v = self.db.execute_command("GRAPH.PASSWORD", "REMOVE", "foo")
        self.db.execute_command("AUTH", "default", "pass")
        self.env.assertTrue(v)
        try:
            v = self.db.execute_command("AUTH", "falkordb-user", "foo")
            self.env.assertTrue(False, "should not be able to authenticate")
        except redis.exceptions.AuthenticationError as e:
            self.env.assertTrue(True)   

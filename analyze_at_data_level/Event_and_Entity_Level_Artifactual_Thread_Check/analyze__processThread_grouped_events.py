import os
import json
from collections import defaultdict
from itertools import combinations

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists for serialization
        return super().default(obj)


if __name__ == "__main__":


   # SET
   processThread_grouped_events_jsons_dirpath = \
      "/data/d1/jgwak1/tabby/graph_embedding_improvement_JY_git/analyze_at_data_level/Event_and_Entity_Level_Artifactual_Thread_Check/RESULTS/Event_and_Entity_Level_Artifactual_Thread_Check__Dataset-Case-1__2024-01-24_134317/processThread_grouped_events"

   intersecting_entities_check = True

   # use_node_identifier == True -- uses entities such as <file-object>__<filename>
   # use_node_identifier == False -- uses entities such as <filename>
   intersecting_entities_check__use_node_identifier = True
   # -----------------------------------------------------------------------------------------------------------------------------------------   

   TO_SAVE_dirpath = os.path.split(processThread_grouped_events_jsons_dirpath)[0]

   if intersecting_entities_check:

         # Log-level analysis 1 : Are there cases, where different threads interact with the same entity?
         #                        (Priti said there was no such case when she checked subgraphs)
         #                        Dobule-check here won log-level.
         #
         #                        For each entity that is not None 
         #                        set-intersection operation across "all thread-1 interaction entitiy" intersect "thread-2 interaction entity" , and so on
         #                        need to think if it makes sense to 

         intersecting_entity_found = dict()

         indices__intersecting_entities__to__pidtid = dict()

         processThread_grouped_events_jsons = os.listdir(processThread_grouped_events_jsons_dirpath)

         cnt = 1

         for processThread_grouped_events_json in processThread_grouped_events_jsons:
            

            index = processThread_grouped_events_json.removesuffix(".json")

            intersecting_entity_found[index] = dict()
            indices__intersecting_entities__to__pidtid[index] = defaultdict(set)

            file_ptr = open( os.path.join(processThread_grouped_events_jsons_dirpath, processThread_grouped_events_json), 'r')
            processThread_grouped_events = json.load(file_ptr)

            processThread_interacted_entity_mapping = defaultdict(set)

            for pid in processThread_grouped_events:
               for tid in processThread_grouped_events[pid]:
                  for log_entry in processThread_grouped_events[pid][tid]:

                     if log_entry['PROVIDER_SPECIFIC_ENTITY'] == None: # this needs to be the first check
                        continue

                     if intersecting_entities_check__use_node_identifier:
                        pass
                     else:
                        if "__" in log_entry['PROVIDER_SPECIFIC_ENTITY']: # drop the object identifie r(as memory)
                           log_entry['PROVIDER_SPECIFIC_ENTITY'] = log_entry['PROVIDER_SPECIFIC_ENTITY'].split("__", 1)[1]

                     if log_entry['PROVIDER_SPECIFIC_ENTITY'] == 'None':
                        continue

                     if log_entry['PROVIDER_SPECIFIC_ENTITY'] == '':
                        continue



                     processThread_interacted_entity_mapping[f"{pid}_{tid}"].add( log_entry['PROVIDER_SPECIFIC_ENTITY'] )


            # how to check if there exists intersecting elements between any two or more sets in list of sets

            pid_tid_keys = list(processThread_interacted_entity_mapping.keys())

            def has_intersection(set1, set2):
               return bool(set1.intersection(set2))

            all_intersecting_entities__in_this_index = set()

            #----------------------------------------------------------------------------------------------------------------------------------------------
            for pid_tid_keys_combination in combinations(pid_tid_keys, 2):
               pid_tid_key1, pid_tid_key2 = pid_tid_keys_combination
               
               set1, set2 = processThread_interacted_entity_mapping[pid_tid_key1], processThread_interacted_entity_mapping[pid_tid_key2]
               
               inteserction = set1.intersection(set2)

               all_intersecting_entities__in_this_index.update(inteserction)
               
               if len(inteserction) > 1:

                  # intersecting_entity_found[index][f"{pid_tid_key1}___AND___{pid_tid_key2}"] = list(inteserction) # since set is not json serializable
                  intersecting_entity_found[index][f"{pid_tid_key1}___AND___{pid_tid_key2}"] = inteserction # since set is not json serializable

            #----------------------------------------------------------------------------------------------------------------------------------------------
            # Now, within this index, get a mapping from intersecting-entity to threads it was part of
            # -- following is inefficient implementation, but that's fine
            

            for intersecting_entity__in_this_index in all_intersecting_entities__in_this_index:

               for matching_pidtid_pair_key in intersecting_entity_found[index]:

                  if intersecting_entity__in_this_index in intersecting_entity_found[index][matching_pidtid_pair_key]:

                        matching_pidtid_pair = matching_pidtid_pair_key.split("___AND___")

                        indices__intersecting_entities__to__pidtid[index][intersecting_entity__in_this_index].update( set(matching_pidtid_pair ) )
                  
                        #print()

                     # indices__intersecting_entities__to__pidtid[index]

            #----------------------------------------------------------------------------------------------------------------------------------------------

                  # print(f"{pid_tid_key1} and {pid_tid_key2} have intersecting elements ({inteserction}).")
            print(f"{cnt}/{len(processThread_grouped_events_jsons)} | {index} got all intersections(if any)", flush=True)
            cnt+=1

         with open(os.path.join(TO_SAVE_dirpath, f"intersecting_entity_found__used_node_identifier_{intersecting_entities_check__use_node_identifier}.json" ), 'w') as json_file:
            json.dump(intersecting_entity_found, json_file,  cls = SetEncoder)


         with open(os.path.join(TO_SAVE_dirpath, f"indices__intersecting_entities__to__pidtid__used_node_identifier_{intersecting_entities_check__use_node_identifier}.json" ), 'w') as json_file:
            json.dump(indices__intersecting_entities__to__pidtid, json_file, cls = SetEncoder)

               #    intersecting_entity_found[processThread_grouped_events_json] = {
               #       ""
               #       "intersecting_entities": result_set
               #    }
               # print()
   # -----------------------------------------------------------------------------------------------------------------------------------------
   # Log-level analysis 2 : Artifactual thread 
         
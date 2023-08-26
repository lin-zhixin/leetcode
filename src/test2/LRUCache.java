package test2;

import java.util.HashMap;
import java.util.Map;

class DLinkNode {
    int key, val;
    DLinkNode pre, next;

    public DLinkNode() {
    }

    public DLinkNode(int key, int val) {
        this.key = key;
        this.val = val;
    }
}

public class LRUCache {
    private DLinkNode head = new DLinkNode(), tail = new DLinkNode();
    Map<Integer, DLinkNode> cache = new HashMap<>();
    int capacity;
    int size;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        size = 0;
        head.next = tail;
        head.pre = tail;
        tail.next = head;
        tail.pre = head;
    }

    public int get(int key) {
        return cache.containsKey(key) && move2Head(cache.get(key)) ? cache.get(key).val : -1;
    }

    public void put(int key, int value) {
        DLinkNode node = null;
        if (cache.containsKey(key)) {
            node = cache.get(key);
            node.val = value;
            move2Head(node);
        } else {
            node = new DLinkNode(key, value);
            addHead(node);
            cache.put(key, node);
            if (size == capacity) {
                cache.remove(remove(tail.pre).key);
            } else {
                size++;
            }
        }

    }

    public DLinkNode addHead(DLinkNode node) {
        node.next = head.next;
        head.next.pre = node;
        head.next = node;
        node.pre = head;
        return node;

    }

    public DLinkNode remove(DLinkNode node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
        return node;
    }

    public boolean move2Head(DLinkNode node) {
        return addHead(remove(node)) != null;
    }

}


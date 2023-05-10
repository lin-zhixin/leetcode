package category;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class LRUCache {
    class DLinkedNode {
        int key;
        int val;
        DLinkedNode pre;
        DLinkedNode next;

        public DLinkedNode() {
        }

        public DLinkedNode(int key, int val) {
            this.key = key;
            this.val = val;
        }
    }

    //双向链表+map
    private Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    DLinkedNode head, tail;
    int capacity;
    int size;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.size = 0;
        this.head = new DLinkedNode();
        this.tail = new DLinkedNode();
        head.next = tail;
        tail.pre = head;
    }

    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if (Objects.nonNull(node)) {
            moveToHead(node);
            return node.val;
        }
        return -1;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if (Objects.nonNull(node)) {
            node.val = value;
            moveToHead(node);
        } else {
            DLinkedNode newNode = new DLinkedNode(key, value);
            addToHead(newNode);
            cache.put(key, newNode);
            size++;
            if (size > capacity) {
                DLinkedNode delNode = deleteTail();
                cache.remove(delNode.key);
                size--;
            }
        }

    }

    private void addToHead(DLinkedNode node) {
        node.next = head.next;
        head.next.pre = node;
        head.next = node;
        node.pre = head;
    }

    private void moveToHead(DLinkedNode node) {
        delete(node);
        addToHead(node);
    }

    private void delete(DLinkedNode node) {
        node.next.pre = node.pre;
        node.pre.next = node.next;
    }

    private DLinkedNode deleteTail() {
        DLinkedNode delNode = tail.pre;
        delete(delNode);
        return delNode;
    }

    public static void main(String[] args) {
        LRUCache lRUCache = new LRUCache(2);
        lRUCache.put(1, 1); // 缓存是 {1=1}
        lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
        System.out.println(lRUCache.get(1));
//        cout <<  << endl;    // 返回 1
        lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
//	lRUCache.displayCache();
        System.out.println(lRUCache.get(2));
//        cout <<  << endl;  // 返回 -1 (未找到)
        lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
//	lRUCache.displayCache();
        System.out.println(lRUCache.get(1));
        System.out.println(lRUCache.get(3));
        System.out.println(lRUCache.get(4));
//        cout << lRUCache.get(1) << endl;    // 返回 -1 (未找到)
//        cout << lRUCache.get(3) << endl;   // 返回 3
//        cout << lRUCache.get(4) << endl;   // 返回 4


    }

}

